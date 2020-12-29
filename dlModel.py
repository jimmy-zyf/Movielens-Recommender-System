import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import tensorflow as tf

from math import sqrt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras import utils
from tensorflow.keras.layers import Input, Reshape, Dot, Add, Activation, Lambda, Concatenate, Dense, Dropout, Embedding
from tensorflow.keras.models import Model, Sequential



class EmbeddingLayer:
    '''
    return a single layer
    '''
    def __init__(self, n_items, n_factors):
        self.n_items = n_items
        self.n_factors = n_factors
    
    def __call__(self, x):
        x = Embedding(self.n_items, self.n_factors, embeddings_initializer='he_normal',
                      embeddings_regularizer=tf.keras.regularizers.L2(1e-6))(x)
        x = Reshape((self.n_factors,))(x)
        return x

# Deep Learning Model
def RecommenderNet(n_users, n_movies, n_factors, min_rating, max_rating):

    user = Input(shape=(1,))
    u = EmbeddingLayer(n_users, n_factors)(user)
    
    movie = Input(shape=(1,))
    m = EmbeddingLayer(n_movies, n_factors)(movie)
    
    x = Concatenate()([u, m])
    x = Dropout(0.05)(x)
    
    x = Dense(10, kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    
    x = Dense(1, kernel_initializer='he_normal')(x)
    x = Activation('sigmoid')(x)
    x = Lambda(lambda x: x * (max_rating - min_rating) + min_rating)(x)
    model = Model(inputs=[user, movie], outputs=x)

    model.compile(loss='mean_squared_error', 
                  optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001),
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model

def dl_train(train_data, test_data,  top_n, n_factor = 30):
    '''
    return a matrix userID and with top k recommendations
    '''

    all_data = pd.concat([train_data,test_data])

    user_enc = LabelEncoder().fit(all_data['userId'].values)
    all_data['userId'] = user_enc.transform(all_data['userId'].values)
    n_users = all_data['userId'].nunique()

    item_enc = LabelEncoder().fit(all_data['movieId'].values)
    all_data['movieId'] = item_enc.transform(all_data['movieId'].values)
    n_movies = all_data['movieId'].nunique()

    all_data['rating'] = all_data['rating'].values.astype(np.float32)
    train_data['rating'] = train_data['rating'].values.astype(np.float32)
    test_data['rating'] = test_data['rating'].values.astype(np.float32)

    min_rating = min(all_data['rating'])
    max_rating = max(all_data['rating'])

    train_data['userId_orig']  = train_data['userId']
    train_data['movieId_orig']  = train_data['movieId']
    
    test_data['userId_orig'] = test_data['userId']
    test_data['movieId_orig'] = test_data['movieId']

    train_data['userId'] = user_enc.transform(train_data['userId'].values)
    train_data['movieId'] = item_enc.transform(train_data['movieId'].values)

    test_data['userId'] = user_enc.transform(test_data['userId'].values)
    test_data['movieId'] = item_enc.transform(test_data['movieId'].values)

    test_movie_id_dict = dict()
    test_user_id_dict = dict()

    test_encodded_mid_list = test_data['movieId'].tolist()
    test_encodded_uid_list = test_data['userId'].tolist()
    test_mid_list = test_data['movieId_orig'].tolist()
    test_uid_list = test_data['userId_orig'].tolist()

    for i,mid in enumerate(test_encodded_mid_list):
      test_movie_id_dict[mid] =  test_mid_list[i]

    for j,uid in enumerate(test_encodded_uid_list):
      test_user_id_dict[uid] =  test_uid_list[j]

    X_train_array = [train_data.iloc[:,0], train_data.iloc[:,1]]
    X_test_array = [test_data.iloc[:,0], test_data.iloc[:,1]]

    y_train = train_data.iloc[:,2]
    y_test = test_data.iloc[:,2]  

    model = RecommenderNet(n_users, n_movies, n_factor, min_rating, max_rating)
    model.fit(x=X_train_array, y=y_train, batch_size=64, epochs=10,
            verbose = False, validation_data=(X_test_array, y_test))
    
    preds = model.predict(X_test_array)
    combined = test_data.copy()
    combined["pred"] = preds
    
    recommend = pd.DataFrame(data = None, columns = ['userId', 'Recommendation']) 
    for uid in test_data.userId.unique():
        selected = combined[combined["userId"] == uid]
        selected = selected.sort_values(by="pred", ascending=False)
        uid_recommendation = []
        uid_movieIds = selected["movieId"].tolist()[:top_n]
        uid_preds = selected["pred"].tolist()[:top_n]
        for i in range(min(top_n,len(selected))):
          uid_recommendation.append((test_movie_id_dict[uid_movieIds[i]],uid_preds[i]))

        new_row = {'userId': test_user_id_dict[uid], 'Recommendation': uid_recommendation}
        recommend = recommend.append(new_row, ignore_index=True)


    return recommend
  

def recommend_single_user(recommend, top_n, single_userId):
    return recommend[recommend['userId'] == single_userId]['Recommendation'].values.tolist()