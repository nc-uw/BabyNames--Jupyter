#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 01:43:57 2018
@author: nc57
"""

import os
import pandas as pd
from glob import glob
import re
import seaborn as sns
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima_model import ARIMA
from collections import defaultdict

'''
create_filt description:
1. the above function summarizes the filtered df
2. distribution of the summarized data is observed
3. appropriate cut-off (n) is selected (in this case number >= 500)
4. the tuple of (name, gender) from this summarized data is then\
     used to filter the original dataframe
5. tuple filter (name, gender) and filtered df is returned
'''

def create_filt(df, n):
    df_grp = df.groupby(['name', 'sex'], as_index=False)['number', 'rank'].mean()
    #df_grp.describe()
    #df_grp.info()
    
    #plot distrbution of mean_number
    print ('\n distribution before filtering \n')
    sns.distplot(df_grp['number'])
    #filter when above plot looks ~normal
    df_grp_filt = df_grp[df_grp['number'] >= n]
    print ('\n distribution after filtering \n')
    sns.distplot(df_grp_filt['number'])
    
    #df_grp_filt.describe()
    #df_grp.info()
    filt = [tuple(x) for x in df_grp_filt[['name', 'sex']].values]
    
    #filter original based on current parameters
    print ('\n data frame info before filtering \n', df.info())
    print ('\n .. filter in progress .. \n')
    df_filt = df[df[['name', 'sex']].apply(tuple, 1).isin(filt)]
    print ('\n data frame info after filtering \n', df_filt.info())
    return df_filt, filt


'''
createX funtion:
1. create appropriate DS for scikit-learn clustering
2. ensure to populate any missing year with 0
i.e. if say obs are present for 1990-2000 and 2006-2009, \
    but not for 2001-2005\
    the value for the missing years should be replaced by 
3. appropriate DS: for a given name, gender a list of numbers corresponding to yearly frequencies
4. shape: unique_name_gender * 20 (time period considered)
'''

def createX(filt, df_filt, mn, mx):
    #df_year is the df with value for all years
    df_year=pd.DataFrame([])
    df_year['year'] = range(mn,mx)
    df_year['number'] = 0
    #X = []
    X = np.array([])
    c=0
    for i in filt:
        #i = filt[209]
        c+=1
        #print tracker
        #print (i, 'i.e.', c, ' of ', len(filt))
        temp = df_filt[(df_filt.name == i[0]) &(df_filt.sex == i[1])][['year', 'number']]
        #df_year is 'outer' merged with the above temp df and na is replaced by 0
        #this ensures all years values are present and data is consistent for modelling
        merge = pd.merge(temp, df_year, how = 'outer', on='year').fillna(0).drop(columns = ['number_y']).rename(columns={'number_x':'number'})
        merge = merge.sort_values(by=['year'])
        #X.append(list(merge['number'].values))
        if c > 1:
            X = np.vstack((X, merge['number'].values))
        else:
            X = merge['number'].values
    return X

#X = createX(filt_90s_00s, df_90s_00s_filt, 1990, 2010)
#print ('\n shape of DS (array):', X.shape)

#sclaing for ARIMA
def scale_ARIMA_X(X, filt_mod):
    F_2020 = []
    scaler = StandardScaler()
    c = 0
    for x in X:
        c+=1
        print('generating forecast for:', filt_mod[c-1], 'i.e. {} of {}'.format(c, len(filt_mod)) )
        xs = scaler.fit_transform(x.reshape(-1,1))
        try:
            model = ARIMA(xs, order=(1,1,1))
            model_fit = model.fit(disp=0)
            f_2020 = model_fit.predict(start=138, end=140)[2]
            f_2020 = scaler.inverse_transform([f_2020])[0]
            print(f_2020)
        except ValueError:
            f_2020 = 'na'
        except np.linalg.LinAlgError:
            f_2020 = 'na'
        finally:
            F_2020.append(f_2020)
    return F_2020

#F_2020 = scale_ARIMA_X(X, filt_mod)
