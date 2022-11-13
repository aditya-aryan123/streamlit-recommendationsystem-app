import streamlit as st
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import pandas as pd
import requests

movies = pd.read_csv('data.csv')
similarity = pickle.load(open('cosine_sim.pkl', 'rb'))

movies_ = movies.reset_index()
indices = pd.Series(movies_.index, index=movies_['title'])

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_['soup'])

cosine_sim = cosine_similarity(tfidf_matrix)

st.title('Movie Recommender System')


def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=7cd46e44ee3774a8cc2fbbc325a28e17".format(movie_id)
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path


def content_recommender(movie):
    movie_index = movies_[movies_['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    recommended_movies = []
    recommended_movie_posters = []

    for i in movies_list:
        movie_id = movies.iloc[i[0]].id
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_movie_posters.append(fetch_poster(movie_id))

    return recommended_movies, recommended_movie_posters


selected_movie_name = st.selectbox(
    "Type or select a movie from the dropdown",
    movies['title'].values
)
if st.button('Show Recommendation'):
    names, posters = content_recommender(selected_movie_name)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(names[0])
        st.image(posters[0])
    with col2:
        st.text(names[1])
        st.image(posters[1])
    with col3:
        st.text(names[2])
        st.image(posters[2])
    with col4:
        st.text(names[3])
        st.image(posters[3])
    with col5:
        st.text(names[4])
        st.image(posters[4])
