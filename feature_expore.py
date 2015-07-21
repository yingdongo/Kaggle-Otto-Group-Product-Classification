# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 09:46:48 2015

@author: Ying
"""

import pandas as pd
from matplotlib import pyplot as plt

def load_data():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    return train,test
train,test=load_data()
train.hist(figsize=(16,12),bins=50)
plt.show()

