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
#gui imports
from tkinter import filedialog
from tkinter import *

# neural net #1
# NUMERICAL FEATURES
def neural_net_numerical_features(url_to_csv, column_to_predict, list_of_features, epochs_amount):
    dataframe = pd.read_csv(url_to_csv)
    dataframe.head()
    #replace nans and infinities in dataframe
    dataframe.replace([np.inf, -np.inf], np.nan).dropna(axis=1)

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
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
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

    return accuracy, prediction_result

# GUI
class popupWindow(object):
    def __init__(self,master):
        top=self.top=Toplevel(master)
        self.choose_button = Button(top, text='Choose CSV file', command=self.choose_file)
        self.choose_button.pack()
        self.l=Label(top,text="Column to predict:", state=DISABLED)
        self.l.pack()
        self.e=Entry(top, state=DISABLED)
        self.e.pack()
        self.l2=Label(top,text="Numerical columns used to predict in a list: (e.g. column1,column2,etc.)", state=DISABLED)
        self.l2.pack()
        self.e2=Entry(top, state=DISABLED)
        self.e2.pack()
        self.l3=Label(top,text="Number of epochs:", state=DISABLED)
        self.l3.pack()
        self.e3=Entry(top, state=DISABLED)
        self.e3.pack()
        self.b=Button(top,text='Ok',command=self.cleanup, state=DISABLED)
        self.b.pack()
        
    def cleanup(self):
        self.value=self.e.get()
        self.value2=self.e2.get()
        self.value3=self.e3.get()
        gui.b2['state'] = 'normal'
        self.top.destroy()
    
    def choose_file(self):
        root.filename = filedialog.askopenfilename(initialdir = '/',title = 'Select file',filetypes = (('csv files','*.csv'),('all files','*.*')))
        if len(str(root.filename)) > 2:
            gui.label_output['text'] = 'File ready.\n'
            self.e['state'] = 'normal'
            self.e2['state'] = 'normal'
            self.e3['state'] = 'normal'
            self.l['state'] = 'normal'
            self.l2['state'] = 'normal'
            self.l3['state'] = 'normal' 
            self.b['state'] = 'normal' 

class mainWindow(object):
    def __init__(self,master):
        self.master=master
        self.b=Button(master,text="Numeric Feature Model",command=self.popup, anchor=CENTER)
        self.b.pack()
        self.b2=Button(master,text="Train",command=self.run_network, state=DISABLED, anchor=CENTER)
        self.b2.pack()
        self.label_output = Label(master, text='')
        self.label_output.pack()

    def popup(self):
        self.w=popupWindow(self.master)
        self.b['state'] = 'disabled' 
        self.master.wait_window(self.w.top)
        self.b['state'] = 'normal'

    def entryValue(self):
        return self.w.value

    def entryValue2(self):
        return self.w.value2

    def entryValue3(self):
        return self.w.value3
    
    def run_network(self):
        accuracy, prediction_result = neural_net_numerical_features(root.filename,str(self.entryValue()),[x.strip() for x in self.entryValue2().split(',')], int(self.entryValue3()))
        self.label_output['text'] += 'Training done. \nModel Accuracy: {}'.format(accuracy)
        self.label_output['text'] += '\nTest Predictions:\n {}'.format(prediction_result)

if __name__ == '__main__': 
    root = Tk()
    gui = mainWindow(root)
    root.geometry('500x200')
    root.mainloop()