from __future__ import absolute_import, division, print_function, unicode_literals
# basic imports
import numpy as np
import csv, sys, os
import pandas as pd
from datetime import date
# tensorflow
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from tensorflow import feature_column
import tensorflow.keras as keras
from tensorflow.keras import *
from sklearn.model_selection import train_test_split

# COLLECTION OF NEURAL NETWORK FUNCTIONS

# Linear regression models are neural networks without activation fucntions,
# therefore linear functions which cant adapt as good

# neural net #1
# NUMERICAL FEATURES TRAINING
def neural_net_numerical_features(url_to_csv, column_to_predict, list_of_features, epochs_amount, optimizer_input, loss_input):
    dataframe = pd.read_csv(url_to_csv)
    dataframe.head()
    # replace nans and infinities in dataframe
    dataframe.replace([np.inf, -np.inf], np.nan).dropna(axis=1)

    train, test = train_test_split(dataframe, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)
    print(len(train), 'train examples')
    print(len(val), 'validation examples')
    print(len(test), 'test examples')

    # A utility method to create a tf.data dataset from a Pandas Dataframe
    def df_to_dataset(dataframe, shuffle=True, batch_size=32):
        dataframe = dataframe.copy()
        # get label column
        labels = dataframe.pop(column_to_predict)
        # to construct a Dataset from data in memory, you can use tf.data.Dataset.from_tensors() or tf.data.Dataset.from_tensor_slices()
        # tf.data.Dataset.from_tensor_slices((data, label))
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
        # shuffle dataset
        if shuffle:
            ds = ds.shuffle(buffer_size=len(dataframe))
        # batch_size: A tf.int64 scalar tf.Tensor, representing the number of consecutive elements of this dataset to combine in a single batch.
        # dataset = dataset.batch(3) 
        # splits data into equal batches (numpy arrays)
        # dataset = dataset.batch(3, drop_remainder=True) drops remainder if no full bacth can be achieved
        ds = ds.batch(batch_size)
        return ds
    
    batch_size = 5 # A small batch sized is used for demonstration purposes
    train_ds = df_to_dataset(train, batch_size=batch_size)
    val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
    test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

    # A utility method to create a feature column
    # and to transform a batch of data
    feature_columns = []

    # choose numeric features
    for header in list_of_features:
        feature_columns.append(feature_column.numeric_column(header))
    
    # layer with features
    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
    
    batch_size = 32
    train_ds = df_to_dataset(train, batch_size=batch_size)
    val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
    test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

    # model
    model = tf.keras.Sequential([
        feature_layer,
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    # optimize the model
    model.compile(optimizer=optimizer_input,
                  loss=loss_input,
                  metrics=['accuracy'])
    
    # train the model
    model.fit(train_ds,
              validation_data=val_ds,
              epochs=epochs_amount)
    
    # model info after training
    loss, accuracy = model.evaluate(test_ds)
    print("Accuracy", accuracy)

    # make predictions for column based on feature columns
    predictions = model.predict(test_ds)
    prediction_result = ''
    for prediction, vars()[column_to_predict] in zip(predictions[:10], list(test_ds)[0][1][:10]): #vars()[column_to_predict] converts the string inputet to a variable
        prediction_result += 'Predicted ' + column_to_predict + ': {:.2%}'.format(prediction[0]) + ' | Actual outcome: ' + ('1' if bool(vars()[column_to_predict]) else '0') + '\n'
    
    model.save('NNCSV_'+str(date.today()), '/') # save model for prediction use later
    model_name = str(date.today())

    return accuracy, prediction_result, model_name

# NUMERICAL FEATURES PREDICTION
def predict_numerical_features(url_to_csv, column_to_predict, model_filename):
    model = keras.models.load_model(model_filename)
    
    
    dataframe = pd.read_csv(url_to_csv)
    #replace nans and infinities in dataframe
    dataframe.replace([np.inf, -np.inf], np.nan).dropna(axis=1)

    def df_to_dataset(dataframe, batch_size=32):
        dataframe = dataframe.copy()
        labels = dataframe.pop(column_to_predict)
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
        ds = ds.batch(batch_size)
        return ds
    
    batch_size = 32
    full_ds = df_to_dataset(dataframe, batch_size=batch_size)
    
    predictions = model.predict(full_ds)
    prediction_result = ''
    for prediction, vars()[column_to_predict] in zip(predictions, list(full_ds)[0][1]): #vars()[column_to_predict] converts the string inputet to a variable
        outcome = 1 if prediction[0] > 0.5 else 0
        prediction_result += 'Predicted ' + column_to_predict + ': {:.2}'.format(prediction[0]) + ' : Predicted Outcome: ' + str(outcome) + '\n'

    return prediction_result

# neural net #2
# WORD FEATURES TRAINING
def neural_net_word_features(url_to_csv, epochs_amount, optimizer_input):
    dataframe = pd.read_csv(url_to_csv)
    dataframe.head()
    # replace nans and infinities in dataframe
    dataframe.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
    
    ds = dataframe.copy()
    labels = ds.label.tolist()
    texts = ds.text.tolist()

    # GET THE LISTS TO BE DATASETS

    # word embedding
    embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
    hub_layer = hub.KerasLayer(embedding, input_shape=[], 
                           dtype=tf.string, trainable=True)
    
    model = tf.keras.Sequential()
    model.add(hub_layer)
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    
    model.compile(optimizer=optimizer_input,
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    history = model.fit(train_dataset.shuffle(10000).batch(512),
                    epochs=epochs_amount,
                    validation_data=validation_dataset.batch(512),
                    verbose=1)
    
    loss, accuracy = model.evaluate(train_dataset)

    model.save('NNCSVT_'+str(date.today()), '/')
    model_name = str(date.today())

    return accuracy, model_name

def get_model_weights(model_filename):
    model = keras.models.load_model(model_filename)
    
    weights = []
    # last layer does not have weigts/biases, so [:-1]
    for count,l in enumerate(model.layers[:-1], 1):
        vars()[str(count)] = model.layers[count].get_weights()[0]
        weights.append(vars()[str(count)])

    return weights

def get_model_biases(model_filename):
    model = keras.models.load_model(model_filename)

    biases = []
    for count,l in enumerate(model.layers[:-1], 1):
        vars()[str(count)] = model.layers[count].get_weights()[1]
        biases.append(vars()[str(count)])

    return biases
