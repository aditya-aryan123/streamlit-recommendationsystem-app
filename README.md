# Movie Recommendation System using Streamlit

Recommendation system that recommends movies similar to those the user likes based on content.

Check out the live demo: https://aditya-aryan123-streamlit-recommendationsystem-app-app-w4qqhu.streamlit.app/


## Overview

The movies are recommended based on the content of the movie you entered or selected. The main parameters that are considered for the recommendations are the genre, overview of the movie, top 10 casts, director, production companies, production countries, tagline, keywords and producer. The details of the movies, such as title, genre, runtime, rating, poster, casts, etc., are fetched from TMDB.


## How to get the API key?

Create an account in https://www.themoviedb.org/. Once you successfully created an account, click on the API link from the left hand sidebar in your account settings and fill all the details to apply for an API key. If you are asked for the website URL, just give "NA" if you don't have one. You will see the API key in your API sidebar once your request has been approved.


## How to run the project?

1. Clone or download this repository to your local machine.
2. Install all the libraries mentioned in the requirements.txt file with the command pip install -r requirements.txt
3. Get your API key from https://www.themoviedb.org/.
4. Replace YOUR_API_KEY in app.py file on line 22 and hit save.
5. Open your terminal/command prompt from your project directory and run the file app.py by executing the command streamlit app.py and it'll automatically open in your browser.


## Similarity Score :

It is a common approach to match similar documents is based on counting the maximum number of common words between the documents.


## How Cosine Similarity works?

Cosine similarity is a metric used to determine how similar the documents are irrespective of their size.
Mathematically, Cosine similarity measures the cosine of the angle between two vectors projected in a multi-dimensional space.
In this context, the two vectors I am talking about are arrays containing the word counts of two documents.
