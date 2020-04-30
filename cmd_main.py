from __future__ import absolute_import, division, print_function, unicode_literals
# basic imports
import numpy as np
import csv, sys, os, datetime
import pandas as pd
from termcolor import colored
# NEURAL NET functions from models.py
from models import *
# plots
import plotly.express as px
# readline for better input line 
try:
    import readline
except:
    pass #readline not available

def main():

    print(colored('[0]', 'green'),'Make Model;           -dataset_filepath -feature_to_predict -list_of_features -epochs -optimizer [adam] -loss_func [binary_crossentropy] -dropout [t/f] -save_model [t/f]')
    print(colored('[1]', 'green'),'Predict model;        -model_filepath -dataset_filepath -feature_to_predict -list_of_features -row_to_predict -nth_element')
    print(colored('[2]', 'green'),'Deep model analysis;  -model_filepath -dataset_filepath -feature_to_predict -list_of_features -nth element -nth_row_element')
    print(colored('[3]', 'green'),'Import analysis data; -dataset_filepath')
    print(colored('[4]', 'green'),'exit script')

    while True:

        i = input('> ')
        i = i.split()

        if str(i[0])=='0':
            i[7]=True if i[7]=='t' else False # convert string t/f to boolean True/False
            i[8]=True if i[8]=='t' else False

            results = neural_net_csv_features(i[1], i[2], i[3].split(','), int(i[4]), i[5], i[6], i[7], i[8])

            for i in results:
                print(i)

        elif str(i[0])=='1':

            results = predict_csv_features(i[1],i[2],i[3],i[4].split(','),int(i[5]),int(i[6]),False,0)

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

        elif str(i[0])=='2':

            results = predict_csv_features(i[1],i[2],i[3],i[4].split(','),0,int(i[5]),True,int(i[6]))
            
            print(f'Rows analysed: {len(results)}')
            print(colored('[0]', 'green'),f'Make importance graph;     -number of feature [1-{int((len(results[1])-1)/2)}]')
            print(colored('[1]', 'green'),f'Make classification graph; -number of feature x -number of feature y [1-{int((len(results[1])-1)/2)}]')
            print(colored('[2]', 'green'),'Save data')
            print(colored('[3]', 'green'),'Exit deep model analyis')

            df = pd.DataFrame(results[0], columns=results[1])
            results[1].remove(results[2]) # remove label column from feature list

            second_loop = True
            while second_loop == True:
                x = input('[Deep model analysis] > ')
                x=x.split()
                num_features=int((len(results[1])-1)/2)

                if str(x[0])=='0':
                    results_hover=results[1].copy()
                    results_hover.remove(results[1][int(x[1])+(num_features-1)])
                    results_hover.remove(results[1][int(x[1])-1])
                    fig = px.scatter(df, x=results[1][int(x[1])+(num_features-1)], y=results[1][int(x[1])-1], color=results[2], color_continuous_scale=px.colors.diverging.Tealrose, hover_name=results[2], hover_data=results_hover)
                    fig.show()
                elif str(x[0])=='1':
                    fig = px.scatter(df, x=results[1][int(x[1])+(num_features-1)], y=results[1][int(x[2])+(num_features-1)], color=results[2], color_continuous_scale=px.colors.diverging.Tealrose, hover_name=results[2])
                    fig.show()
                elif str(x[0])=='2':
                    df.to_csv(f'{date.today()}_{datetime.datetime.now().time()}.csv')
                elif str(x[0])=='3':
                    second_loop=False
                else:
                    print('Invalid Command.')

        elif str(i[0])=='3':
            df = pd.read_csv(i[1])
            results=list(df.columns.values)
            column_to_predict=results[-1]
            results.remove(results[-1])
            results.remove(results[0])

            print(colored('[0]', 'green'),f'Make importance graph;     -number of feature [1-{int((len(results)-1)/2)}]')
            print(colored('[1]', 'green'),f'Make classification graph; -number of feature x -number of feature y [1-{int((len(results)-1)/2)}]')
            print(colored('[2]', 'green'),'Exit deep model analyis')

            second_loop = True
            while second_loop == True:
                x = input('[Deep model analysis] > ')
                x=x.split()
                num_features=int((len(results)-1)/2)

                if str(x[0])=='0':
                    #fig = px.parallel_coordinates(df, color=results[2], labels=results[1],
                    #    color_continuous_scale=px.colors.diverging.Tealrose, color_continuous_midpoint=0.5)
                    results_hover=results.copy()
                    results_hover.remove(results[int(x[1])+(num_features-1)])
                    results_hover.remove(results[int(x[1])-1])
                    fig = px.scatter(df, x=results[int(x[1])+(num_features-1)], y=results[int(x[1])-1], color=column_to_predict, color_continuous_scale=px.colors.diverging.Tealrose, hover_name=column_to_predict, hover_data=results_hover)
                    fig.show()
                elif str(x[0])=='1':
                    fig = px.scatter(df, x=results[int(x[1])+(num_features-1)], y=results[int(x[2])+(num_features-1)], color=column_to_predict, color_continuous_scale=px.colors.diverging.Tealrose, hover_name=column_to_predict)
                    fig.show()
                elif str(x[0])=='2':
                    second_loop=False
                else:
                    print('Invalid Command.')

        elif str(i[0])=='4':
            sys.exit()
        else:
            print('Invalid Command.')

if __name__ == '__main__':
    main()

