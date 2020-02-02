from __future__ import absolute_import, division, print_function, unicode_literals
# basic imports
import numpy as np
import csv, sys, os
import pandas as pd
from datetime import date
#gui imports
from tkinter import filedialog
from tkinter import *
#NEURAL NET functions from models.py
from models import*

# GUI

class popupWindow(object):
    def __init__(self,master):
        top=self.top=Toplevel(master)
        self.choose_button = Button(top, text='Choose CSV file', command=self.choose_file)
        self.choose_button.pack()
        self.l=Label(top,text='Column to predict:', state=DISABLED)
        self.l.pack()
        self.e=Entry(top, state=DISABLED)
        self.e.pack()
        self.l2=Label(top,text='Numerical columns used to predict\n in a list: (e.g. column1,column2,etc.)', state=DISABLED)
        self.l2.pack()
        self.e2=Entry(top, state=DISABLED)
        self.e2.pack()
        self.l3=Label(top,text='Number of epochs:', state=DISABLED)
        self.l3.pack()
        self.e3=Entry(top, state=DISABLED)
        self.e3.pack()
        
        self.l4=Label(top,text='Optimizer:', state=DISABLED)
        self.l4.pack()
        options = ['SGD','RMSprop','Adagrad','Adadelta','Adam','Adamax','Nadam']
        variable = StringVar(top)
        variable.set(options[4]) # default value
        self.w = OptionMenu(top, variable, *options)
        self.w.configure(state="disabled")
        self.w.pack()

        self.b=Button(top,text='Save',command=self.cleanup, state=DISABLED)
        self.b.pack(padx=5, side=RIGHT, anchor=S)
        self.b2=Button(top,text='Train',command=self.run_network, state=DISABLED)
        self.b2.pack(padx=5, side=RIGHT, anchor=S)
        self.b3=Button(top,text='Clear Output',command=self.clear_output)
        self.b3.pack(padx=5, side=LEFT, anchor=S)
        self.label_output = Label(top, text='')
        self.label_output.pack()

    def run_network(self):
        accuracy, prediction_result, model_name = neural_net_numerical_features(root.filename,str(self.entryValue()),[x.strip() for x in self.entryValue2().split(',')], int(self.entryValue3()), optimizer_input)
        self.label_output['text'] += 'Training done. \nModel Accuracy: {}'.format(accuracy)
        self.label_output['text'] += '\nTest Predictions:\n {}'.format(prediction_result)
        self.label_output['text'] += '\nModel saved in folder:\n {}'.format(model_name)
        
    def entryValue(self):
        return self.value

    def entryValue2(self):
        return self.value2

    def entryValue3(self):
        return self.value3
    
    def clear_output(self):
        self.label_output['text'] = ''

    def cleanup(self):
        self.value=self.e.get()
        self.value2=self.e2.get()
        self.value3=self.e3.get()
        self.b2['state'] = 'normal'
    
    def choose_file(self):
        root.filename = filedialog.askopenfilename(initialdir = '/',title = 'Select file',filetypes = (('csv files','*.csv'),('all files','*.*')))
        if len(str(root.filename)) > 2:
            self.label_output['text'] = 'File ready.\n'
            self.e['state'] = 'normal'
            self.e2['state'] = 'normal'
            self.e3['state'] = 'normal'
            self.l['state'] = 'normal'
            self.l2['state'] = 'normal'
            self.l3['state'] = 'normal'
            self.w['state'] = 'normal'
            self.l4['state'] = 'normal' 
            self.b['state'] = 'normal' 

class popupWindow_predict(object):
    def __init__(self,master):
        top=self.top=Toplevel(master)
        self.choose_button_predict = Button(top, text='Choose CSV file to predict', command=self.choose_predict_file)
        self.choose_button_predict.pack()
        self.choose_model_button_predict = Button(top, text='Choose model folder for prediction', command=self.choose_model_file)
        self.choose_model_button_predict.pack()
        self.l_predict=Label(top,text='Column to predict:', state=DISABLED)
        self.l_predict.pack()
        self.e_predict=Entry(top, state=DISABLED)
        self.e_predict.pack()
        self.b_predict=Button(top,text='Train',command=self.run_predict, state=DISABLED)
        self.b_predict.pack()
        self.label_output_predict=Label(top,text='', state=DISABLED)
        self.label_output_predict.pack()
        
    def choose_predict_file(self):
        global predict_filename
        predict_filename = filedialog.askopenfilename(initialdir = '/',title = 'Select file',filetypes = (('csv files','*.csv'),('all files','*.*')))
        print(predict_filename)
        if len(str(predict_filename)) > 2:
            self.label_output_predict['text'] = 'Predict File ready.\n'
            self.e_predict['state'] = 'normal'
            self.l_predict['state'] = 'normal'
            self.b_predict['state'] = 'normal'
            self.label_output_predict['state'] = 'normal'
    
    def entryValue_predict(self):
        return self.e_predict.get()

    def choose_model_file(self):
        global model_filename
        model_filename = filedialog.askdirectory(initialdir = os.getcwd())
        print(model_filename)
        
    def run_predict(self):
        results = predict_numerical_features(predict_filename, self.entryValue_predict(), model_filename)
        self.label_output_predict['text'] = 'Predictions:\n{}'.format(results)

class mainWindow(object):
    def __init__(self,master):
        self.master=master
        self.label1=Label(master,text='Numeric Models')
        self.label1.config(font=('Sans-serif', 12))
        self.label1.pack(pady=5,padx=5)
        self.b=Button(master,text='Numeric Feature Model Train',command=self.popup, anchor=CENTER)
        self.b.pack(padx=20)
        self.label2=Label(master,text='Train a model with a csv file\n and save it in a file for later predictions.')
        self.label2.pack(pady=5)
        self.b_p=Button(master,text='Numeric Feature Model Predict',command=self.popup_predict, anchor=CENTER)
        self.b_p.pack(padx=20)
        self.label3=Label(master,text='Predict columns in a csv file with a saved model.')
        self.label3.pack(pady=5)
        #self.b4=Button(master,text='Predict on new dataset',command=self.predict_new, anchor=SE)
        #self.b4.pack()

    def popup(self):
        self.w=popupWindow(self.master)
        self.b['state'] = 'disabled' 
        self.master.wait_window(self.w.top)
        self.b['state'] = 'normal'

    def popup_predict(self):
        self.w=popupWindow_predict(self.master)
        self.b['state'] = 'disabled' 
        self.master.wait_window(self.w.top)
        self.b['state'] = 'normal'

if __name__ == '__main__': 
    root = Tk(className='ml_models')
    gui = mainWindow(root)
    root.mainloop()