from __future__ import absolute_import, division, print_function, unicode_literals
# basic imports
import numpy as np
import csv, sys, os
import pandas as pd
import datetime
from termcolor import colored
# NEURAL NET functions from models.py
from models import *
# plots
import plotly.express as px 
try:
    import readline
except:
    pass #readline not available

def main():
    print(colored('[0]', 'green'),'Predict model;       -model_filepath -dataset_filepath -feature_to_predict -row_to_predict -nth element')
    print(colored('[1]', 'green'),'Deep model analysis; -model_filepath -dataset_filepath -feature_to_predict -nth element -nth row element')
    print(colored('[2]', 'green'),'exit script')
    while True:
        i = input('> ')
        i = i.split()
        if str(i[0])=='0':
            results = predict_csv_features(i[1],i[2],i[3],int(i[4]),int(i[5]),False,0)
            for c,i in enumerate(results):
                if c==0:
                    print(f'Prediction: {round(float(i[0]), 3)} |',colored(f'{i[1]}', attrs=['bold']))
                    continue
                if i[1]>=0.5:
                    print(colored(i[0], 'green'),f' [{round(i[1], 3)}]')
                elif 0.5>i[1]>=0.1:
                    print(colored(i[0], 'yellow'),f' [{round(i[1], 3)}]')
                elif 0.1>i[1]>=0.0:
                    print(colored(i[0], 'red'),f' [{round(i[1], 3)}]')
        elif str(i[0])=='1':
            results = predict_csv_features(i[1],i[2],i[3],0,int(i[4]),True,int(i[5]))
            print('Rows analysed: ',colored(f'{len(results)}', attrs=['bold']))
            print(colored('[0]', 'green'),'Make graphs')
            print(colored('[1]', 'green'),'Save data')
            print(colored('[2]', 'green'),'Exit deep model analyis')
            second_loop=True
            while second_loop==True:
                x = input('[Deep model analysis] > ')

                df = pd.DataFrame(results[0], columns=results[1])
                results[1].remove(results[2])

                if str(x)=='0':
                    #fig = px.parallel_coordinates(df, color=results[2], labels=results[1],
                    #    color_continuous_scale=px.colors.diverging.Tealrose, color_continuous_midpoint=0.5)
                    
                    fig = px.scatter(df, x=results[1][3], y=results[1][1], color=results[2], color_continuous_scale=px.colors.diverging.Tealrose)
                    fig.show()
                elif str(x)=='1':
                    df.to_csv(f'{date.today()}_{datetime.datetime.now().time()}.csv')
                elif str(x)=='2':
                    second_loop=False
                    break
                else:
                    pass

        elif str(i[0])=='2':
            sys.exit()
        else:
            print('Invalid Command.')

if __name__ == '__main__':
    main()

