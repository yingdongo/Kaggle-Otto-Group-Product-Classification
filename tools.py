# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 00:49:25 2015

@author: Ying
"""

import pandas as pd
from sklearn import cross_validation
import numpy as np
from sklearn.metrics import log_loss
pd.set_option('chained_assignment',None)

def load_data(path):
    return pd.read_csv(path)

def multiclass_log_loss(y_true, y_pred, eps=1e-15):
   
    predictions = np.clip(y_pred, eps, 1 - eps)

    # normalize row sums to 1
    predictions /= predictions.sum(axis=1)[:, np.newaxis]

    actual = np.zeros(y_pred.shape)
    rows = actual.shape[0]
    actual[np.arange(rows), y_true.astype(int)] = 1
    vsota = np.sum(actual * np.log(predictions))
    return -1.0 / rows * vsota

def cv_score(clf,x_train,y):
    score=cross_validation.cross_val_score(clf, x_train, y, scoring='log_loss',
                                           cv=5, n_jobs=5, verbose=0, fit_params=None, score_func=None, 
                                           pre_dispatch='2*n_jobs')                                      
    return score.mean()

def cv_score1(clf,x_train,y):
    score=cross_validation.cross_val_score(clf, x_train, y, scoring=None,
                                           cv=5, n_jobs=16, verbose=0, fit_params=None, score_func=None, 
                                           pre_dispatch='2*n_jobs')                                      
    return score.mean()

def split_data(data,cols):
    X=data[cols]
    y=data['target']
    return X,y
    
def cross_v(clf,training,target):
    skf = cross_validation.StratifiedKFold(target, n_folds=10, random_state=42,shuffle=True)
   
    results = []
    for train_index, test_index in skf:
        X_train, y = training[train_index], target[train_index]
        fit= clf.fit(X_train, y)
        probas=fit.predict_proba(training[test_index])
        results.append(multiclass_log_loss(target[test_index], probas) )
    #print out the mean of the cross-validated results
    print "Results: " + str( np.array(results).mean() )

    return np.array(results).mean()
