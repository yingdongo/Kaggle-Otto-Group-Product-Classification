# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 09:29:19 2015

@author: Ying
"""

from sklearn.decomposition import PCA
from tools import load_data
from numpy import corrcoef, sum, log, arange
from pylab import pcolor, show, colorbar, xticks, yticks
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
#==============================================================================
# train=load_data('train.csv')
# cols = [col for col in train.columns if col not in ['id','target']] 
# x_data=train[cols]
# fig, ax = plt.subplots()
# R = corrcoef(x_data.T)
#==============================================================================
#==============================================================================
# pcolor(R)
# colorbar()
# ax.set_xticks(np.arange(x_data.shape[1]) + 0.5, minor=False) #code to align the labels with the correct column of data on plot
# ax.set_yticks(np.arange(x_data.shape[1]) + 0.5, minor=False)#code to align the labels with the correct row of data on plot
# ax.set_xticklabels(x_data.columns, rotation=90)
# ax.set_yticklabels(x_data.columns)
# ax.set_title("Correlation Matrix of Features")
#==============================================================================

#==============================================================================
# pca = PCA(n_components=None)
# pca_X = pca.fit_transform(x_data)
# label = [("pca_" + str(e)) for e in range(pca_X.shape[1])]
# train1=pd.DataFrame(data=pca_X,columns=label)
#==============================================================================

def feature_engineering(data):    
    mean = np.mean(data, axis = 1)
    std = np.std(data, axis = 1)
    nonzero=0
    for col in data.columns:
        n=data[col]>0
        nonzero=nonzero+n
    data['nonzero']=nonzero
    data['mean'] = mean
    data['std'] = std
    return data

