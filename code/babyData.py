#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 19:03:58 2018
@author: nc57
"""

import os
import pandas as pd
import chardet
import matplotlib.pyplot as plt
from glob import glob
import re
import unicodedata
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.arima_model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np

def get_data():
    #specify folder where data was dloaded
    path = os.getcwd()
    folder = path + '/names'
    #move inside folder which has .txt files
    os.chdir(folder)
    #retrieve names of .txt files only
    files = sorted(glob(os.path.join('*.txt')))

    #check filenames and count
    print ('\ntotal files present in folder {}: {}'.format(folder, len(files)))
    print ('\nfirst file name', files[0])
    print ('\nlast file name', files[-1])
    #os.chdir(path)
    return files

#read filenames to dataframe
def df_create(f, col = ['name', 'sex', 'number']):
    df = pd.read_csv(f, header=None, names = col)
    df['year'] = int(re.findall(r'\d+', f)[0])
    df['rank'] = list(df.index)
    return df

def read_files(files):
    #fetch files
    #files = get_data()
    #inital df
    df = df_create(files[0])
    #loop thru rest of data and concat with initial df
    for f in files[1:]:
        df_temp = df_create(f)
        print ('appending: ', f)
        df = pd.concat([df, df_temp])
    return df
 
