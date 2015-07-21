# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 11:47:17 2015

@author: Ying
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from tools import load_data
from tools import split_data
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import LinearSVC
from feature_engineering import feature_engineering
from tools import cv_score
from tools import cross_v
from tools import cv_score1
import pandas as pd
from sklearn import preprocessing
from sklearn import ensemble
plt.style.use('ggplot')

def create_clf():
    forest=ensemble.RandomForestClassifier(n_estimators=100)
    return forest

def plot_importances(importances, col_array):
# Calculate the feature ranking
    indices = np.argsort(importances)[::-1]    
#Mean Feature Importance
    #print "\nMean Feature Importance %.6f" %np.mean(importances)    
#Plot the feature importances of the forest
    plt.figure(figsize=(20,8))
    plt.title("Feature importances")
    plt.bar(range(len(importances)), importances[indices],
            color="gr", align="center")
    plt.xticks(range(len(importances)), col_array[indices], fontsize=14, rotation=90)
    plt.xlim([-1, len(importances)])
    plt.show()

def select_rfecv():
    data=load_data('train.csv')
    train=data.loc[np.random.choice(data.index,np.around(len(data)*0.5), replace=False)]
    feature_cols= [col for col in train.columns if col  not in ['target','id']]
    X_train,y=split_data(train,feature_cols)
    estimator = SVC(kernel="linear")
    selector = RFECV(estimator, step=1, cv=5)
    selector = selector.fit(X_train, y)
    print selector.support_ 
    print selector.ranking_
    print selector.grid_scores_
    print feature_cols
#all pass...
    
def get_score(data,cols):
    clf = RandomForestClassifier(n_estimators=100)
    X_train,y_train=split_data(data,cols)
    return cv_score1(clf,X_train,y_train)
    
def select():
    train=load_data('train.csv')
    le = preprocessing.LabelEncoder()
    le.fit(train['target'])
    train['target']=le.transform(train['target'])
    cols = [col for col in train.columns if col not in  ['id','target']] 
    X_train=feature_engineering(train[cols])
    cols = [col for col in X_train.columns if col not in  ['id','target']] 
    cols1=['nonzero','mean','std']
    print 'total'
    cross_v(create_clf(),X_train.values,train['target'].values)
    scores=np.array(np.zeros(93))
    for e,col in enumerate(list(cols1)):      
          feature_cols=list(cols)    
          feature_cols.remove(col)
          print col 
          score=cross_v(create_clf(),X_train[cols].values,train['target'].values)
          scores[e]=score
    return pd.DataFrame(data=scores,index=cols)
#'feat_80','feat_20','feat_82','feat_45','feat_6','feat_10','feat_41','feat_33','feat_57','feat_21'

def select_feature(X_train,y,feature_cols,importances):
    indices = np.argsort(importances)[::-1]    
    f_count=len(importances)+1
    f_start=20
    f_range=range(f_start,f_count)
    score=np.array(np.zeros(f_count-f_start))
    for i in f_range:
        cols=feature_cols[indices]
        cols=cols[:i]
        print i        
        cv=cross_v(create_clf(),X_train[cols].values,y.values)
        score[i-f_start]=cv
    return pd.DataFrame(score,index=f_range)

def select_tree(X,y):
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X,y)
    X_new=clf.transform(X)
    return X_new
    #no use
    
def select_variance(X):
    sel = VarianceThreshold(threshold=(.7 * (1 - .7)))
    X=sel.fit_transform(X)
    return X
    #two eliminated
    
def select_basedonimportance():
    train = load_data('train.csv')
    feature_cols= [col for col in train.columns if col  not in ['target','id']]
    y_train = train['target']
    X_train=feature_engineering(train[feature_cols])
    clf=create_clf()
    clf.fit(X_train, y_train)
    importances=clf.feature_importances_
    scores=select_feature(X_train,y_train,X_train.columns,importances)
    plt.figure(figsize=(20,8))
    plt.plot(np.abs(scores))
    plt.xticks(range(len(scores)),scores.index)
    plt.show()
    
#select_basedonimportance()#94 84 95 6 96 82
scores=select()

#X_new=select_tree(X_train, y_train)

#X_new = LinearSVC(C=0.01, penalty="l1", dual=False).fit_transform(X_train, y_train)
#print X_new.shape 
#all pass
#train = load_data('train.csv')
#train=train.loc[np.random.choice(train.index,np.around(len(train)*1), replace=False)]
#feature_cols= [col for col in train.columns if col  not in ['target','id']]
#X_train = train[feature_cols]
#y_train = train['target']
#X_train=feature_engineering(X_train)

#X=select_variance(X_train)


#clf = RandomForestClassifier()
#clf.fit(X_train, y_train)
#plot_importances(clf.feature_importances_,X_train.columns)
#==============================================================================
