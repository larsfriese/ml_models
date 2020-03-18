from __future__ import absolute_import, division, print_function, unicode_literals
# basic imports
import numpy as np
import csv, sys, os
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
# tensorflow
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from tensorflow import feature_column
import tensorflow.keras as keras
from tensorflow.keras import *
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

def i_outputs(model, dataset, layer_index):
    layer_names = []
    for layer in model.layers:
        layer_names.append(layer.name)
    model.summary()
    intermediate_layer_model = Model(inputs=model.input,
                                    outputs=model.get_layer(layer_names[layer_index]).output)
    intermediate_output = intermediate_layer_model.predict(dataset)

    return intermediate_output[0]
    
# COLLECTION OF NEURAL NETWORK FUNCTIONS

# Linear regression models are neural networks without activation functions,
# therefore linear functions which cant adapt as good

# neural net #1
# NUMERICAL/TEXT FEATURES TRAINING
def neural_net_csv_features(url_to_csv, column_to_predict, list_of_features_numeric, list_of_features_word, epochs_amount, optimizer_input, loss_input, dropout, save_model):
            
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
    feature_layer_inputs = {}
    
    all_words = []
    if list_of_features_word[0] is not '': 
        # find all words
        for word_column in list_of_features_word:
            for i in dataframe[word_column].tolist():
                splitted = i.split()
                for x in splitted:
                    if x not in all_words: all_words.append(x)

            text = feature_column.categorical_column_with_vocabulary_list(
                word_column, all_words)
            text_embedding = feature_column.embedding_column(text, dimension=8)
            feature_columns.append(text_embedding)
            feature_layer_inputs[word_column] = tf.keras.Input(shape=(1,), name=word_column, dtype=tf.string)

    if list_of_features_numeric[0] is not '':
        # choose numeric features
        for header in list_of_features_numeric:
            feature_columns.append(feature_column.numeric_column(header))
            feature_layer_inputs[header] = tf.keras.Input(shape=(1,), name=header)
        
    # layer with features
    batch_size = 32
    feature_layer = tf.keras.layers.DenseFeatures(feature_columns, name='fl')#, input_shape=(batch_size,))

    train_ds = df_to_dataset(train, batch_size=batch_size)
    val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
    test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

    # layers
    # no difference between acivation functions and layers
    '''relu = tf.keras.layers.LeakyReLU(alpha=0.1)

    # model
    model = tf.keras.Sequential()

    if len(dataframe.index)<100:
        #model.add(layers.InputLayer(input_shape=(0,), name='0'))
        model.add(feature_layer)
        model.add(layers.Dense(64, activation=relu, name='1'))
        model.add(layers.Dense(64, activation=relu, name='2'))
        model.add(layers.Dense(1, activation='sigmoid'))
    elif 100<len(dataframe.index)<1000:
        model.add(feature_layer)
        model.add(layers.Dense(128, activation=relu, name='1'))
        model.add(layers.Dense(128, activation=relu, name='2'))
        model.add(layers.Dense(1, activation='sigmoid'))
    elif 1000<len(dataframe.index):
        model.add(feature_layer)
        model.add(layers.Dense(128, activation=relu, name='1'))
        model.add(layers.Dense(128, activation=relu, name='2'))
        model.add(layers.Dense(128, activation=relu, name='3'))
        model.add(layers.Dense(1, activation='sigmoid'))

    '''

    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
    feature_layer_outputs = feature_layer(feature_layer_inputs)

    x = layers.Dense(128, activation='relu')(feature_layer_outputs)
    x = layers.Dense(64, activation='relu')(x)

    baggage_pred = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs=[v for v in feature_layer_inputs.values()], outputs=baggage_pred)

    cb_list=[EarlyStopping(monitor='accuracy', min_delta=0.005, patience=10, baseline=None, mode='auto')] if dropout==True else []
    # optimize the model
    model.compile(optimizer=optimizer_input,
                  loss=loss_input,
                  metrics=['accuracy'])

    # train the model
    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=epochs_amount,
                        callbacks=cb_list)

    last_acc = history.history['accuracy'][-1]
    last_val_acc = history.history['val_accuracy'][-1]
    
    all_accs = 0
    for i in history.history['accuracy']: all_accs += i
    avg_acc = all_accs/(len(history.history['accuracy']))

    val_all_accs = 0
    for i in history.history['val_accuracy']: val_all_accs += i
    avg_val_acc = val_all_accs/(len(history.history['val_accuracy']))

    # overfitting
    if (avg_acc-avg_val_acc) > 0.3:
        accuracy = 'Warning: The Model might be overfitting, as the training accuracy is\n {:.2f} and the validation accuracy is {:.2f}.\n Possible solutions:\n - use more training data\n - remove irrelevant features, add more relevant features.'.format(avg_acc, avg_val_acc)
    else:
        accuracy = 'Training acc.: {:.2f} Test acc.: {:.2f}'.format(avg_acc, avg_val_acc)

    # make predictions for column based on feature columns
    predictions = model.predict(test_ds)
    prediction_result = ''
    for prediction, vars()[column_to_predict] in zip(predictions[:10], list(test_ds)[0][1][:10]): #vars()[column_to_predict] converts the string inputet to a variable
        prediction_result += 'Predicted ' + column_to_predict + ': {:.2%}'.format(prediction[0]) + ' | Actual outcome: ' + ('1' if bool(vars()[column_to_predict]) else '0') + '\n'
    
    # TEST AREA
    
    print(i_outputs(model, val_ds, -2))

    if save_model==True:
        model.save('NNCSV_'+str(date.today()), '/') # save model for prediction use later
        model_info = '\nModel saved in folder:\n {}\n\n'.format(str(date.today()))
    else:
        model_info = ''

    return accuracy, prediction_result, model_info

# NUMERICAL/TEXT FEATURES PREDICTION
def predict_csv_features(url_to_csv, column_to_predict, model_filename):
    model = keras.models.load_model(model_filename) #custom_objects={'LeakyReLU': tf.keras.layers.LeakyReLU}
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

# get weights and biases of hidden layers of any model
def get_model_weights(model_filename):
    model = keras.models.load_model(model_filename)
    weights = []
    # last layer does not have weigts/biases, so [:-1]
    for count,l in enumerate(model.layers[:-1], 1):
        l = model.layers[count].get_weights()[0]
        weights.append(l)

    return weights

def get_model_biases(model_filename):
    model = keras.models.load_model(model_filename)
    biases = []
    for count,l in enumerate(model.layers[:-1], 1):
        l = model.layers[count].get_weights()[1]
        biases.append(l)

    return biases
