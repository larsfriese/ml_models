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
from models import *

class CreateToolTip(object):
    '''
    create a tooltip for a given widget
    '''
    def __init__(self, widget, text='widget info'):
        self.widget = widget
        self.text = text
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.close)
    def enter(self, event=None):
        x = y = 0
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        # creates a toplevel window
        self.tw = tkinter.Toplevel(self.widget)
        # Leaves only the label and removes the app window
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry("+%d+%d" % (x, y))
        label = tkinter.Label(self.tw, text=self.text, justify='left',
                       background='yellow', relief='solid', borderwidth=1,
                       font=("times", "8", "normal"))
        label.pack(ipadx=1)
    #def close(self, event=None):
    #    if self.tw:
    #        self.tw.destroy()

# GUI

class popupWindow(object):
    def __init__(self,master):
        top=self.top=Toplevel(master)
        self.choose_button = Button(top, text='Choose CSV file', command=self.choose_file)
        self.choose_button.grid(row=0, column=3)
        self.fillall = Button(top, text='Add all columns', command=self.add_all_columns, state=DISABLED)
        self.fillall.grid(row=1, column=3)
        self.l_options=Label(top,text='', wraplengt=400)
        self.l_options.grid(row=0, columnspan=3)
        self.l=Label(top,text='Column to predict:', state=DISABLED)
        self.l.grid(row=1)
        self.e=Entry(top, state=DISABLED)
        self.e.grid(row=1, column=2)
        self.l2=Label(top,text='Numerical columns used to predict\n in a list: (e.g. column1,column2,etc.)', state=DISABLED)
        self.l2.grid(row=2)
        self.e2=Entry(top, state=DISABLED)
        self.e2.grid(row=2, column=2)
        self.l6=Label(top,text='Text columns used to predict\n in a list: (e.g. column1,column2,etc.)', state=DISABLED)
        self.l6.grid(row=3)
        self.e4=Entry(top, state=DISABLED)
        self.e4.grid(row=3, column=2)
        self.l3=Label(top,text='Number of epochs:', state=DISABLED)
        self.l3.grid(row=4)
        self.e3=Entry(top, state=DISABLED)
        self.e3.grid(row=4, column=2)
        self.dp_var = IntVar()
        self.ch=Checkbutton(top, text='Dropout', state=DISABLED, variable=self.dp_var)
        self.ch.grid(row=4, column=3)
        #self.ttp_ch = CreateToolTip(self.ch, 'Training stops when theres no change over 0.005 in acc over 10 steps.')
        
        self.l4=Label(top,text='Optimizer:', state=DISABLED)
        self.l4.grid(row=5)
        options = ['sgd','rmsprop','adagrad','adadelta','adam','adamax','nadam']
        self.optimizer = StringVar(top)
        self.optimizer.set(options[4]) # default value
        self.w = OptionMenu(top, self.optimizer, *options)
        self.w.configure(state="disabled")
        self.w.grid(row=5, column=2)

        self.l5=Label(top,text='Loss:', state=DISABLED)
        self.l5.grid(row=6)
        options_loss = ['mean_squared_error','mean_squared_logarithmic_error','mean_absolute_error','binary_crossentropy','hinge','squared_hinge','categorical_crossentropy','sparse_categorical_crossentropy','kullback_leibler_divergence']
        self.loss = StringVar(top)
        self.loss.set(options_loss[3]) # default value
        self.w2 = OptionMenu(top, self.loss, *options_loss)
        self.w2.configure(state="disabled")
        self.w2.grid(row=6, column=2)

        self.b2=Button(top,text='Train',command=self.run_network, state=DISABLED)
        self.b2.grid(row=8, column=3)
        self.b3=Button(top,text='Clear Output',command=self.clear_output)
        self.b3.grid(row=11, column=3)
        self.save_var = IntVar()
        self.ch2=Checkbutton(top, text='Save Model', state=DISABLED, variable=self.save_var)
        self.ch2.grid(row=9, column=3)
        self.analysis_var = IntVar()
        self.ch3=Checkbutton(top, text='Model Analysis', state=DISABLED, variable=self.analysis_var)
        self.ch3.grid(row=10, column=3)
        self.label_output = Label(top, text='')
        self.label_output.grid(row=8, pady=5, padx=5, column=0, columnspan=3, rowspan=2)

    def run_network(self):
        self.value=self.e.get()
        self.value2=self.e2.get()
        self.value3=self.e3.get()
        self.value4=self.e4.get()
        accuracy, prediction_result, model_info = neural_net_csv_features(root.filename,str(self.entryValue()),[x.strip() for x in self.entryValue2().split(',')],[x.strip() for x in self.entryValue4().split(',')], int(self.entryValue3()), self.optimizer.get(), self.loss.get(), self.dp_var.get(), self.save_var.get(), self.analysis_var.get())
        self.label_output['text'] = ''
        self.label_output['text'] += 'Training done. \n{}'.format(accuracy)
        self.label_output['text'] += '\nTest Predictions:\n {}'.format(prediction_result)
        self.label_output['text'] += model_info
        
    def entryValue(self):
        return self.value

    def entryValue2(self):
        return self.value2

    def entryValue3(self):
        return self.value3
    
    def entryValue4(self):
        return self.value4
    
    def clear_output(self):
        self.label_output['text'] = ''
    
    def add_all_columns(self):
        dataframe = pd.read_csv(root.filename)
        for i in list(dataframe.columns):
            first_column = dataframe[i].iloc[1]
            if self.e.get() == list(dataframe.columns)[list(dataframe.columns).index(i)]:
                continue
            if type(first_column)==str:
                self.e4.insert(END, str(i)+',')
            else:
                self.e2.insert(END, str(i)+',')
        self.e2.delete(len(self.e2.get())-1, END)
        self.e4.delete(len(self.e4.get())-1, END)

    def choose_file(self):
        root.filename = filedialog.askopenfilename(initialdir = os.getcwd(),title = 'Select file',filetypes = (('csv files','*.csv'),('all files','*.*')))
        dataframe = pd.read_csv(root.filename)
        options_str = ''
        if len(list(dataframe.columns))>30:
            for i in list(dataframe.columns[:30]):
                if i==list(dataframe.columns)[30]:
                    options_str += i
                else:
                    options_str += i + ', '
            options_str += ' ...'
        else:
            for i in list(dataframe.columns):
                if i==list(dataframe.columns):
                    options_str += i
                else:
                    options_str += i + ', '
            options_str = options_str[1:-2]
        self.l_options['text'] = 'Columns: ' + options_str
        if len(str(root.filename)) > 2:
            self.label_output['text'] = 'File ready.\n'
            self.e['state'] = 'normal'
            self.e2['state'] = 'normal'
            self.e3['state'] = 'normal'
            self.l['state'] = 'normal'
            self.l2['state'] = 'normal'
            self.l3['state'] = 'normal'
            self.w['state'] = 'normal'
            self.l5['state'] = 'normal'
            self.w2['state'] = 'normal'
            self.l4['state'] = 'normal'
            self.b2['state'] = 'normal'
            self.e4['state'] = 'normal'
            self.l6['state'] = 'normal'
            self.ch['state'] = 'normal'
            self.ch2['state'] = 'normal'
            self.ch3['state'] = 'normal'
            self.fillall['state'] = 'normal'

class popupWindow_predict(object):
    def __init__(self,master,model_fn):
        global model_filename
        model_filename=model_fn
        analysis_filename=None
        top=self.top=Toplevel(master)
        self.choose_button_predict = Button(top, text='Choose CSV file to predict', command=self.choose_predict_file)
        self.choose_button_predict.grid(row=1)
        self.choose_model_button_predict = Button(top, text='Choose model folder for prediction', command=self.choose_model_file)
        self.choose_model_button_predict.grid(row=2)
        self.l_predict_filename=Label(top,text='', wraplengt=200)
        self.l_predict_filename.grid(row=1, column=1)
        self.l_model_filename=Label(top,text=model_filename, wraplengt=200)
        self.l_model_filename.grid(row=2, column=1)
        self.choose_analysis_button_predict = Button(top, text='Choose analysis file for prediction', command=self.choose_analysis_file)
        self.choose_analysis_button_predict.grid(row=3)
        self.l_analysis_filename=Label(top,text='', wraplengt=200)
        self.l_analysis_filename.grid(row=3, column=1)
        self.l_predict=Label(top,text='Feature to predict:', state=DISABLED)
        self.l_predict.grid(row=4)
        self.e_predict=Entry(top, state=DISABLED)
        self.e_predict.grid(row=4, column=1)
        self.l2_predict=Label(top,text='Row to predict (which test sample) for analysis:', state=DISABLED)
        self.l2_predict.grid(row=5)
        self.e2_predict=Entry(top, state=DISABLED)
        self.e2_predict.grid(row=5, column=1)
        self.b_predict=Button(top,text='Predict',command=self.run_predict, state=DISABLED)
        self.b_predict.grid(row=6, column=1)
        self.label_output_predict=Label(top,text='', state=DISABLED)
        self.label_output_predict.grid(row=7, columnspan=2)

    def choose_predict_file(self):
        global predict_filename
        predict_filename = filedialog.askopenfilename(initialdir = os.getcwd(),title = 'Select file',filetypes = (('csv files','*.csv'),('all files','*.*')))
        if len(str(predict_filename)) > 2:
            self.label_output_predict['text'] = 'Predict File ready.\n'
            self.e_predict['state'] = 'normal'
            self.l_predict['state'] = 'normal'
            self.e2_predict['state'] = 'normal'
            self.l2_predict['state'] = 'normal'
            self.b_predict['state'] = 'normal'
            self.choose_analysis_button_predict['state'] = 'normal'
            self.label_output_predict['state'] = 'normal'
            self.l_predict_filename['text'] = predict_filename
    
    def entryValue_predict(self):
        return self.e_predict.get()

    def entryValue_predict2(self):
        return self.e2_predict.get()

    def choose_model_file(self):
        global model_filename
        model_filename = filedialog.askdirectory(initialdir = os.getcwd())
        self.l_model_filename['text']=model_filename
        self.b_weights['state'] = 'normal'
    
    def choose_analysis_file(self):
        global analysis_filename
        analysis_filename = filedialog.askopenfilename(initialdir = os.getcwd(),title = 'Select file',filetypes = (('csv files','*.csv'),('all files','*.*')))
        self.l_analysis_filename['text']=analysis_filename
        
    def run_predict(self):
        global model_filename
        global analysis_filename
        results = predict_csv_features(predict_filename, self.entryValue_predict(), model_filename, analysis_filename, self.entryValue_predict2())
        self.label_output_predict['text'] = 'Predictions:\n{}'.format(results)

class mainWindow(object):
    def __init__(self,master):
        self.master=master
        self.b=Button(master,text='CSV Model Train',command=self.popup, anchor=CENTER)
        self.b.grid(row=2)
        self.label2=Label(master,text='Make Models:')
        self.label2.grid(row=1)
        self.b_p=Button(master,text='CSV Model Predict',command=self.popup_predict, anchor=CENTER)
        self.b_p.grid(row=2, column=1)
        self.label3=Label(master,text='Predict with saved models:')
        self.label3.grid(row=1, column=1)
        self.label3=Label(master,text='Saved models in cwd:')
        self.label3.grid(row=1, column=3)
        self.lb1 = Listbox(master, width=25)
        self.lb1.grid(row=5, column=3, sticky='se')
        self.lb1.bind('<Double-Button>', self.onselect)
        self.b3=Button(master,text='Refresh',command=self.refresh, anchor=CENTER).grid(row=2, column=3, sticky='e')
        
        # adding found files to list
        global dirs_found
        dirs_found = []
        for count,dr in enumerate(os.listdir(os.getcwd()),1):
            if 'ml_analysis' in dr:
                dirs_found.append(dr)
                self.lb1.insert(count, 'CSV Analysis: '+ dr)   
            if 'ml_model' in dr:
                dirs_found.append(dr)
                self.lb1.insert(count, 'CSV Model: '+ dr)   
    
    # refreshing the list
    def refresh(self):
        global dirs_found
        dirs_found = []
        self.lb1.delete(0,'end')
        for count,dr in enumerate(os.listdir(os.getcwd()),1):
            if 'ml_analysis' in dr:
                dirs_found.append(dr)
                self.lb1.insert(count, 'CSV Analysis: '+ dr)   
            if 'ml_model' in dr:
                dirs_found.append(dr)
                self.lb1.insert(count, 'CSV Model: '+ dr)   

    def popup(self):
        self.w=popupWindow(self.master)
        self.b['state'] = 'disabled' 
        self.b_p['state'] = 'disabled'
        self.master.wait_window(self.w.top)
        self.b['state'] = 'normal'
        self.b_p['state'] = 'normal'

    def popup_predict(self):
        self.w=popupWindow_predict(self.master,'')
        self.b['state'] = 'disabled' 
        self.b_p['state'] = 'disabled'
        self.master.wait_window(self.w.top)
        self.b['state'] = 'normal'
        self.b_p['state'] = 'normal'
    
    def popup_text(self):
        self.w=popupWindow_csv_text(self.master)
        self.b['state'] = 'disabled' 
        self.b_p['state'] = 'disabled'
        self.master.wait_window(self.w.top)
        self.b['state'] = 'normal'
        self.b_p['state'] = 'normal'

    def onselect(self, evt):
        global dirs_found
        w = evt.widget
        index = int(w.curselection()[0])
        value = w.get(index)
        dir_name=dirs_found[index]

        self.w=popupWindow_predict(self.master, dir_name)
        self.b['state'] = 'disabled' 
        self.b_p['state'] = 'disabled'
        self.master.wait_window(self.w.top)
        self.b['state'] = 'normal'
        self.b_p['state'] = 'normal'

if __name__ == '__main__': 
    root = Tk(className='ml_models')
    root.resizable(False, False)
    gui = mainWindow(root)
    root.mainloop()