import re
import ast
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

credits_df = pd.read_csv('credits.csv')
keywords_df = pd.read_csv('keywords.csv')
movies_metadata = pd.read_csv('movies_metadata.csv', low_memory=False)
ratings = pd.read_csv('ratings_small.csv')

movies = movies_metadata.loc[(movies_metadata['status'] == 'Released') & (movies_metadata['vote_count'] >= 100)]
movies = movies.loc[(movies['runtime'] >= 45) & (movies['runtime'] <= 300)]


def clean_ids(x):
    try:
        return int(x)
    except:
        return np.nan


movies['id'] = movies['id'].apply(clean_ids)
movies = movies[movies['id'].notnull()]

credits_df['id'] = credits_df['id'].apply(clean_ids)
credits_df = credits_df[credits_df['id'].notnull()]

keywords_df['id'] = keywords_df['id'].apply(clean_ids)
keywords_df = keywords_df[keywords_df['id'].notnull()]

merged_df = credits_df.merge(movies, on='id')
merged_df = merged_df.merge(keywords_df, on='id')


def weighted_rating(x):
    C = merged_df['vote_average'].mean()
    m = merged_df['vote_count'].quantile(0.80)
    v = x['vote_count']
    R = x['vote_average']
    return (v / (v + m) * R) + (m / (m + v) * C)


merged_df['score'] = merged_df.apply(weighted_rating, axis=1)

merged_df.fillna('', axis=1, inplace=True)

final_data = merged_df[['cast', 'crew', 'id', 'genres', 'title', 'overview',
                        'production_companies', 'production_countries', 'spoken_languages',
                        'tagline', 'keywords', 'score', 'popularity']]

data = final_data.copy()

data['genres'] = data['genres'].fillna('[]')
data['genres'] = data['genres'].apply(ast.literal_eval)
data['genres'] = data['genres'].apply(lambda x: [i['name'].lower() for i in x] if isinstance(x, list) else [])

features = ['cast', 'crew', 'keywords', 'production_companies', 'production_countries', 'spoken_languages']
for feature in features:
    data[feature] = data[feature].apply(ast.literal_eval)


def get_director(x):
    for crew_member in x:
        if crew_member['job'] == 'Director':
            return crew_member['name']
    return np.nan


def get_producer(x):
    for crew_member in x:
        if crew_member['job'] == 'Producer':
            return crew_member['name']
    return np.nan


data['director'] = data['crew'].apply(get_director)
data['producer'] = data['crew'].apply(get_producer)


def generate_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        if len(names) > 3:
            names = names[:10]
        return names
    return []


data['cast'] = data['cast'].apply(generate_list)
data['keywords'] = data['keywords'].apply(generate_list)
data['production_countries'] = data['production_countries'].apply(generate_list)
data['production_companies'] = data['production_companies'].apply(generate_list)
data['spoken_languages'] = data['spoken_languages'].apply(generate_list)

data['tagline'] = data['tagline'].str.lower()
data['tagline'] = data['tagline'].apply(lambda x: x.split(' '))

data.drop('crew', axis=1, inplace=True)

data['clean_overview'] = data['overview'].str.lower()
data['clean_overview'] = data['clean_overview'].apply(lambda x: re.sub('[^a-zA-Z]', ' ', x))
data['clean_overview'] = data['clean_overview'].apply(lambda x: re.sub('\s+', ' ', x))
data['clean_overview'] = data['clean_overview'].apply(lambda x: nltk.word_tokenize(x))
data['clean_overview']

stop_words = nltk.corpus.stopwords.words('english')
plot = []
for sentence in data['clean_overview']:
    temp = []
    for word in sentence:
        if word not in stop_words and len(word) >= 3:
            temp.append(word)
    plot.append(temp)

data['clean_overview'] = plot


def sanitize(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''


for feature in ['keywords', 'spoken_languages', 'director', 'producer',
                'overview', 'production_companies', 'cast', 'genres', 'tagline']:
    data[feature] = data[feature].apply(sanitize)


def create_soup(x):
    return ' '.join(x['cast']) + ' ' + ' '.join(x['genres']) + ' ' + ' '.join(
        x['overview']) + ' ' + ' '.join(x['production_companies']) + ' ' + ' '.join(
        x['spoken_languages']) + ' ' + ' ' + ' '.join(x['tagline']) + ' ' + ' '.join(x['keywords']) + ' ' + x[
               'director'] + ' ' + x['producer']


data['soup'] = data.apply(create_soup, axis=1)


def stemming(content):
    port_stem = PorterStemmer()
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


data['soup'] = data['soup'].apply(stemming)

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['soup'])

cosine_sim = cosine_similarity(tfidf_matrix)

data = data.reset_index()
indices = pd.Series(data.index, index=data['title'])

data.to_csv('data.csv', index=False)


def content_recommender(title, cosine_sim=cosine_sim, df=data, indices=indices):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices]


pickle.dump(cosine_sim, open('cosine_sim.pkl', 'wb'))
