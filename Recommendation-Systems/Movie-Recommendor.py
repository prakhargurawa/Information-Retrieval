# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 12:12:48 2020

@author: prakh

https://www.datacamp.com/community/tutorials/recommender-systems-python
"""

"""
This dataset consists of the following files:

1. movies_metadata.csv: This file contains information on ~45,000 movies featured in the Full MovieLens dataset.
   Features include posters, backdrops, budget, genre, revenue, release dates, languages, production countries, and companies.
2. keywords.csv: Contains the movie plot keywords for our MovieLens movies. Available in the form of a stringified 
   JSON Object.
3. credits.csv: Consists of Cast and Crew Information for all the movies. Available in the form of 
   a stringified JSON Object.
4. links.csv: This file contains the TMDB and IMDB IDs of all the movies featured in the Full MovieLens dataset.
5. links_small.csv: Contains the TMDB and IMDB IDs of a small subset of 9,000 movies of the Full Dataset.
6. ratings_small.csv: The subset of 100,000 ratings from 700 users on 9,000 movies.
"""

# DATASET : https://www.kaggle.com/rounakbanik/the-movies-dataset/data

# Import Pandas
import pandas as pd

# Load Movies Metadata
metadata = pd.read_csv('movies_metadata.csv', low_memory=False)

# Print the first three rows
metadata.head(3)

# Calculate mean of vote average column
C = metadata['vote_average'].mean()
print(C)

# Calculate the minimum number of votes required to be in the chart, m
m = metadata['vote_count'].quantile(0.90)
print(m)

# Filter out all qualified movies into a new DataFrame
q_movies = metadata.copy().loc[metadata['vote_count'] >= m]
q_movies.shape


# Function that computes the weighted rating of each movie
def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)


# Define a new feature 'score' and calculate its value with `weighted_rating()`
q_movies['score'] = q_movies.apply(weighted_rating, axis=1)


#Sort movies based on score calculated above
q_movies = q_movies.sort_values('score', ascending=False)

#Print the top 15 movies
q_movies[['title', 'vote_count', 'vote_average', 'score']].head(20)


# CONTENT BASED RECOMMENDATION
#Print plot overviews of the first 5 movies.
metadata['overview'].head()


#Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
metadata['overview'] = metadata['overview'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(metadata['overview'])

#Output the shape of tfidf_matrix
tfidf_matrix.shape

#Array mapping from feature integer indices to feature name.
tfidf.get_feature_names()[5000:5010]

# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

cosine_sim.shape

cosine_sim[1]

#Construct a reverse map of indices and movie titles
indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()

indices[:10]

# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return metadata['title'].iloc[movie_indices]

print(get_recommendations('The Dark Knight Rises'))
















































