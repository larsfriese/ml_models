from __future__ import absolute_import, division, print_function, unicode_literals
# basic imports
import numpy as np
import csv, sys, os, time, math
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
from collections import Counter
# tensorflow
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from tensorflow import feature_column
import tensorflow.keras as keras
from tensorflow.keras import *
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

##########################################
# COLLECTION OF NEURAL NETWORK FUNCTIONS #
##########################################

# NUMERICAL/TEXT FEATURES TRAINING (BINARY CLASSIFICATION)
def neural_net_csv_features(url_to_csv, column_to_predict, list_of_features, epochs_amount, optimizer_input, loss_input, dropout, save_model):

    # Input format: url_to_csv[str], column_to_predict[str], list_of_features[list of str], epochs_amount[int], optimizer_input[str], loss_input[str], dropout[boolean], save_model[boolean]

    global neurons_reviewed, dense_layers

    dataframe = pd.read_csv(url_to_csv)
    dataframe.head()
    # replace nans and infinities in dataframe
    dataframe.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
    
    list_of_features_numeric=[]
    list_of_features_word=[]
    for i in list_of_features:
        first_column = dataframe[i].iloc[1]
        if type(first_column)==str:
            list_of_features_word.append(i)
        else:
            list_of_features_numeric.append(i)

    train, test = train_test_split(dataframe, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)

    # a utility method to create a tf.data dataset from a pandas dataframe
    def df_to_dataset(dataframe, shuffle=True, batch_size=32):
        dataframe = dataframe.copy()
        labels = dataframe.pop(column_to_predict)
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(dataframe))
        ds = ds.batch(batch_size)
        return ds
    
    batch_size = 5
    train_ds = df_to_dataset(train, batch_size=batch_size)
    val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
    test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

    feature_columns = []
    
    all_words = []
    if list_of_features_word is not []: 
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

    if list_of_features_numeric is not []:
        # choose numeric features
        for header in list_of_features_numeric:
            feature_columns.append(feature_column.numeric_column(header))
        
    batch_size = 32
    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

    train_ds = df_to_dataset(train, batch_size=batch_size)
    val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
    test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

    # layers
    # no difference between acivation functions and layers
    relu = tf.keras.layers.LeakyReLU(alpha=0.1)

    model = tf.keras.Sequential()
    bias=True
    
    if len(dataframe.index)<1000:
        model.add(feature_layer)
        model.add(layers.Dense(64, activation=relu))
        model.add(layers.Dense(64, activation=relu))
        model.add(layers.Dense(1, activation='sigmoid'))
        dense_layers=2
    elif 1000<=len(dataframe.index)<10000:
        model.add(feature_layer)
        model.add(layers.Dense(128, activation=relu))
        model.add(layers.Dense(128, activation=relu))
        model.add(layers.Dense(1, activation='sigmoid'))
        dense_layers=2
    elif 10000<=len(dataframe.index):
        model.add(feature_layer)
        model.add(layers.Dense(128, activation=relu))
        model.add(layers.Dense(128, activation=relu))
        model.add(layers.Dense(128, activation=relu))
        model.add(layers.Dense(1, activation='sigmoid'))
        dense_layers=3

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
    
    # get average acc over training
    all_accs = 0
    for i in history.history['accuracy']: all_accs += i
    avg_acc = all_accs/(len(history.history['accuracy']))

    val_all_accs = 0
    for i in history.history['val_accuracy']: val_all_accs += i
    avg_val_acc = val_all_accs/(len(history.history['val_accuracy']))

    # overfitting
    if (avg_acc-avg_val_acc) > 0.3:
        accuracy = '\nWarning: The Model might be overfitting, as the training accuracy is\n {:.2f} and the validation accuracy is {:.2f}.\n Possible solutions:\n - use more training data\n - remove irrelevant features, add more relevant features.'.format(avg_acc, avg_val_acc)
    else:
        accuracy = '\nTraining acc.: {:.2f} Test acc.: {:.2f}'.format(avg_acc, avg_val_acc)

    if save_model==True:
        model.save(os.getcwd() + 'ml_model_'+str(date.today()), '/') # save model for prediction use later
        model_info = 'Model saved in folder: {}\n'.format(str(date.today()))
    else:
        model_info = ''
    
    print(model.summary())

    # Output Format: [accuracy[str], model_info[str]]

    return [accuracy, model_info]

# NUMERICAL/TEXT FEATURES PREDICTION/ANALYSIS (BINARY CLASSIFICATION)
def predict_csv_features(model_filename, url_to_csv, column_to_predict, features, row_in_dataset, nth, deep_analysis, nth_deep):

    # Input format: model_filename[str], url_to_csv[str], column_to_predict[str], features[list of str], row_in_dataset[int], nth[int], deep_analysis[boolean], nth_deep[int]

    model = keras.models.load_model(model_filename) #custom_objects={'LeakyReLU': tf.keras.layers.LeakyReLU}
    dataframe = pd.read_csv(url_to_csv)
    #replace nans and infinities in dataframe
    dataframe.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
    full_dataframe = dataframe.copy()

    batch_size = 32
    def df_to_dataset(dataframe, batch_size=32):
        dataframe = dataframe.copy()
        labels = dataframe.pop(column_to_predict)
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
        ds = ds.batch(batch_size)
        return ds

    if deep_analysis==True:
        importance_deep=[]

        full_ds = df_to_dataset(full_dataframe, batch_size=batch_size)
        predictions = model.predict(full_ds, batch_size=batch_size)
        predictions = predictions.tolist()
        for i in predictions:
            if i[0]>=0.5:
                predictions[predictions.index(i)]=1
            else:
                predictions[predictions.index(i)]=0
        full_dataframe[column_to_predict]=predictions

        t0 = time.time()
        for z in range(1, len(full_dataframe.index)+1):
            if z % nth_deep is not 0:
                continue

            mse = tf.keras.losses.MeanSquaredError()
            importance=[]
            
            for i in features:
                true_output_for_feature = full_dataframe[column_to_predict].iloc[int(z)-1]
                org_ds = df_to_dataset(full_dataframe, batch_size=batch_size)
                predictions_org = model.predict(org_ds, batch_size=batch_size)
                # eorig = L(y, f(X)) (e.g. mean squared error)
                error_org = mse(true_output_for_feature, predictions_org[int(z)-1])

                error_perm = 0
                list_values = full_dataframe[i].tolist()
                l = []
                for e in range(len(full_dataframe.index)):
                    l.append(int(e))
                l.remove(int(z)-1)

                df = full_dataframe.copy()
                df = df.drop(l)

                for c, x in enumerate(list_values):
                    if c % nth is not 0:
                        continue
                    # Xperm
                    df.at[int(z)-1, i] = x
                    perm_ds = df_to_dataset(df, batch_size=batch_size)
                    predictions_perm = model.predict(perm_ds, batch_size=batch_size)
                    # eperm = L(Y,f(Xperm))
                    error = mse(true_output_for_feature, predictions_perm[0])
                    error_perm += error.numpy()
                    counter = float(((((list_values.index(x)/len(list_values))*((1/len(features)))+(features.index(i)/len(features))))*(1/len(dataframe.index))+(z/len(dataframe.index)))*100)
                    #if counter % 5 == 0:
                    print(f'{counter}%')
                    
                error_p = error_perm/(len(list_values))
                importance.append(abs(error_p-error_org.numpy()))

            #print(((t1-t0)*(len(full_dataframe.index)/nth_deep))+((t2-t1)*(len(features))*(len(full_dataframe.index)/nth_deep))+((t3-t2)*(len(list_values)/nth)*(len(features)))))
            t5 = time.time()

            for i in features:
                importance.append(full_dataframe[i].iloc[int(z-1)])
            importance.append(z)
            importance.append(true_output_for_feature)
            importance_deep.append(importance)
            prediction_result = []

        print(f'Process took: {round(t5-t0, 2)}s')

    else:
        mse = tf.keras.losses.MeanSquaredError()
        importance=[]

        l = []
        for i in range(len(dataframe.index)):
            l.append(int(i))
        l.remove(int(row_in_dataset)-1)
        dataframe = dataframe.drop(dataframe.index[l])
        
        full_ds = df_to_dataset(dataframe, batch_size=batch_size)

        predictions = model.predict(full_ds, batch_size=batch_size)
        prediction_result = []
        for prediction, vars()[column_to_predict] in zip(predictions, list(full_ds)[0][1]): #vars()[column_to_predict] converts the string inputet to a variable
            outcome = 1 if prediction[0] > 0.5 else 0
            prediction_result.append([prediction[0],outcome])
        
        t0 = time.time()
        for i in features:
            true_output_for_feature = full_dataframe[column_to_predict].iloc[int(row_in_dataset)-1]
            org_ds = df_to_dataset(full_dataframe, batch_size=batch_size)
            predictions_org = model.predict(org_ds, batch_size=batch_size)
            # eorig = L(y, f(X)) (e.g. mean squared error)
            error_org = mse(true_output_for_feature, predictions_org[int(row_in_dataset)-1])

            error_perm = 0
            list_values = full_dataframe[i].tolist()
            l = []
            for e in range(len(full_dataframe.index)):
                l.append(int(e))
            l.remove(int(row_in_dataset)-1)

            df = full_dataframe.copy()
            df = df.drop(l)

            for c, x in enumerate(list_values):
                if c % nth is not 0:
                    continue
                # Xperm
                df.at[int(row_in_dataset)-1, i] = x
                perm_ds = df_to_dataset(df, batch_size=batch_size)
                predictions_perm = model.predict(perm_ds, batch_size=batch_size)
                # eperm = L(Y,f(Xperm))
                error = mse(true_output_for_feature, predictions_perm[0])
                error_perm += error.numpy()
                counter = float(((list_values.index(x)/len(list_values))*((1/len(features)))+(features.index(i)/len(features)))*100)
                if counter % 10 == 0:
                    print(f'{counter}%')
            
            error_p = error_perm/(len(list_values))

            # FIj= eperm - eorig
            importance.append([i, abs(error_p-error_org.numpy())])
        # Sort features by descending FI
        importance.sort(key=lambda x: x[1])
        importance = list(reversed(importance))
        t1 = time.time()

        print(f'Process took: {round(t1-t0, 2)}s')
    
    if deep_analysis==True:
        prediction_result.extend(importance_deep)
        features_values=[]
        for i in features:
            features_values.append(i+'_val')
        for i in features:
            features[features.index(i)]=i+'_imp'
        features.extend(features_values)
        features.append('row_number')
        features.append(column_to_predict)

        # Output Format for Deep Analysis/Prediction (deep_analysis=True): 
        # [[[importance of feature[float], row_number[int], predicted outcome label column[int(1 or 0)]]],
        # [feature[str], 'row_number', column_to_predict[str]],
        # column_to_predict[str]]

        return [prediction_result, features, column_to_predict]
    else:
        prediction_result.extend(importance)

        # Output Format for Normal Analysis/Prediction (deep_analysis=False):
        # [[name of feature[str], importance[float]]]

        return prediction_result