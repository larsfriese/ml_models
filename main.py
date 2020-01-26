from __future__ import absolute_import, division, print_function, unicode_literals
# basic imports
import numpy as np
import csv, sys
import pandas as pd
# tensorflow
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
# additional packages
from twitter_scraper import get_tweets

# collect data for processing #1 (via twitter)
def twitter_scrapers(hashtag_var):

    tweets = []
    for tweet in get_tweets(hashtag_var, pages=5):
        tweets.append(tweet)
    print(str(len(tweets)) + ' tweets succesfully scraped.')

    return tweets

# neural net #1
# NUMERICAL FEATURES
def neural_net_numerical_features(url_to_csv, column_to_predict):
    dataframe = pd.read_csv(url_to_csv)
    dataframe.head()

    train, test = train_test_split(dataframe, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)
    print(len(train), 'train examples')
    print(len(val), 'validation examples')
    print(len(test), 'test examples')

    # A utility method to create a tf.data dataset from a Pandas Dataframe
    def df_to_dataset(dataframe, shuffle=True, batch_size=32):
        dataframe = dataframe.copy()
        labels = dataframe.pop(column_to_predict)
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(dataframe))
        ds = ds.batch(batch_size)
        return ds

    batch_size = 32
    train_ds = df_to_dataset(train, batch_size=batch_size)
    val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
    test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

    example_batch = next(iter(train_ds))[0]

    # A utility method to create a feature column
    # and to transform a batch of data
    feature_columns = []

    # choose numeric features
    for header in ['temperaturemin','temperaturemax','precipitation','mist','fastest2minwindspeed']:
        feature_columns.append(feature_column.numeric_column(header))
    
    # layer with features
    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
    
    # model
    model = tf.keras.Sequential([
        feature_layer,
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    # optimize the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    # train the model
    model.fit(train_ds,
              validation_data=val_ds,
              epochs=8)
    
    # model info after training
    loss, accuracy = model.evaluate(test_ds)
    print("Accuracy", accuracy)
    
    # make predictions for column based on feature columns
    predictions = model.predict(test_ds)
    for prediction, vars()[column_to_predict] in zip(predictions[:10], list(test_ds)[0][1][:10]):
        print('Predicted ' + column_to_predict + ': {:.2%}'.format(prediction[0]),
              " | Actual outcome: ",
              ("1" if bool(vars()[column_to_predict]) else "0"))

if __name__ == '__main__': 
    neural_net_numerical_features('datasets/weather_dataset.csv','fog')
    
    