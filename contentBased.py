from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

from math import sqrt

import numpy as np
import pandas as pd
import random

import wordcloud
from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def content_based_train(train_data, test_data, movies_df, top_n):
  # parse movies genres
  movies_df['genres'] = movies_df['genres'].str.split('|').fillna("").astype('str')
  
  # merge movies genre with train data
  movie_train = movies_df[movies_df['movieId'].isin(train_data['movieId'])]
  
  # build movie profile based on genre
  tfvec = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), stop_words='english')
  genre_tfidf_matrix = tfvec.fit_transform(movie_train['genres'])
  genre_tfidf_matrix = genre_tfidf_matrix.todense()
  genre_df = pd.DataFrame(genre_tfidf_matrix)
  genre_df.insert(loc = 0, column = 'movieId', value = movie_train['movieId'].tolist())
  genre_df.set_index('movieId', inplace = True)
  
  # compute weight
  train_data['sum_weight'] = train_data.groupby('userId')['rating'].transform(lambda x: x.sum())
  train_data['weight'] = train_data['rating'] / train_data['sum_weight']
  
  # build user profile
  weighted_genre_user = {}
  for user in train_data.userId.unique():
    weight = train_data[train_data['userId'] == user]['weight']
    movie = train_data[train_data['userId'] == user]['movieId'].values.tolist()
    new_genre = np.transpose(genre_df[genre_df.index.isin(movie)].values).dot(weight.values.reshape(-1, 1))
    new_genre = [item for elem in new_genre for item in elem]
    weighted_genre_user[user] = new_genre
  genre_user = pd.DataFrame(weighted_genre_user).T
  genre_movie = genre_df

  # make recommendation for test_data
  recommend = pd.DataFrame(data = None, columns = ['userId', 'Recommendation']) 
  for userID in test_data.userId.unique():
    result = []
    for idx in genre_movie.index:
        cos_sim = cosine_similarity([genre_movie.loc[idx].tolist()], [genre_user.loc[userID].tolist()])
        result.append((idx, cos_sim.item((0, 0)))) 
    recommendation = [(item[0], item[1] * 5) for item in sorted(result, key = lambda x: x[1], reverse= True)[:top_n]]
    new_row = {'userId': userID, 'Recommendation': recommendation}
    recommend = recommend.append(new_row, ignore_index=True)
  return recommend

def recommend_single_user(recommend, top_n, single_userId):
  return recommend[recommend['userId'] == single_userId]['Recommendation'].values.tolist()