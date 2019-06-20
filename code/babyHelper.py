#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 20:18:51 2018

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
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

##outlier analysis
'''
create_filt description:
1. the above function summarizes the filtered df
2. distribution of the summarized data is observed
3. appropriate cut-off (n) is selected (in this case number >= 500) on summarized data
4. the tuple of (name, gender) from this summarized data is then\
     used to filter the original dataframe
5. tuple filter (name, gender) and filtered df is returned
'''

#filter based on >=1990 and <2010
#df_90s_00s = df[(df.year >= 1990) & (df.year <= 2009)]
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


#df_90s_00s_filt, filt_90s_00s = create_filt(df_90s_00s, 500)
#Prepare for clustering
'''
1. create appropriate DS for scikit-learn clustering
2. ensure to populate any missing year with 0
i.e. if say obs are present for 1990-2000 and 2006-2009, \
    but not for 2001-2005\
    the value for the missing years should be replaced by 0
3. appropriate DS: for a given name, gender a list of numbers corresponding to yearly frequencies
4. shape: unique_name_gender * 20 (time period considered)
'''

#df_year is the df with value for all years
def merging( df_90s_00s_filt, filt_90s_00s):
    df_year=pd.DataFrame([])
    df_year['year'] = range(1990,2010)
    df_year['number'] = 0
    X = []
    c=0
    for i in filt_90s_00s:
        c+=1
        #print tracker
        #print (i, 'i.e.', c, ' of ', len(filt_90s_00s))
        temp = df_90s_00s_filt[(df_90s_00s_filt.name == i[0]) &(df_90s_00s_filt.sex == i[1])][['year', 'number']]
        #df_year is 'outer' merged with the above temp df and na is replaced by 0
        #this ensures all years values are present and data is consistent for modelling
        merge = pd.merge(temp, df_year, how = 'outer', on='year').fillna(0).drop(columns = ['number_y']).rename(columns={'number_x':'number'})
        X.append(list(merge['number'].values))

    X = np.array(X)    
    print ('\n shape of DS (array):', X.shape)
    return X

#function for plotting and retrieval
def plot_findings(i,X, labels):
    plt.plot(np.arange(1990,2010), np.mean(X[labels==i], axis =0), label=i)
    plt.title('Mean of number (cluster {})'.format(i))
    plt.xlabel("Year")
    plt.ylabel("Frequency")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),\
               fancybox=True, shadow=True, ncol=10)
    plt.show()

def print_findings(i,label_dict):
    print ('\nnames in cluser {}\n'.format(i))
    for key, value in label_dict.items():
        if i == value:
            print (key)
    
