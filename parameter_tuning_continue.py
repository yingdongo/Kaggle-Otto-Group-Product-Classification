# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 00:28:47 2015

@author: Ying
"""
from sklearn.grid_search import GridSearchCV
from sklearn import ensemble
#from feature_engineering import feature_engineering
from tools import load_data
from tools import split_data
from sklearn.grid_search import ParameterGrid
from tools import cv_score1
from sklearn import preprocessing
import numpy as np
def get_clfs():
    return {
            'extra trees' : { 
                'est' :ensemble.ExtraTreesClassifier(),
                'grid' : {
                    'n_estimators' : [500,700],
                    'max_features': [.5,0.8,None ]
                }
            },
#0.820776366398
#{'max_features': 0.5, 'n_estimators': 500}
            'gbrt' : { 
                'est' :ensemble.GradientBoostingClassifier(),
                'grid' : {
                    'n_estimators' : [500,700],
                    'learning_rate': [0.1,0.03],
                }
            },
#0.804857946281
#{'n_estimators': 500, 'learning_rate': 0.1}
            'random forests' : { 
                'est' :ensemble.RandomForestClassifier(),
                'grid' : {
                    'n_estimators' : [500,700],
                    'max_features': [0.03, .1, .5, None ],
                }
            },
#0.811322279324
#{'max_features': 0.1, 'n_estimators': 500}
        }
def get_extra_trees():
    return {
            'extra trees' : { 
                'est' :ensemble.ExtraTreesClassifier(),
                'grid' : {
                    'n_estimators' : [600,700,800],
                    'max_features': [.7,.8,.9,None ]
                }
            }
            }#[0.8210353176616414, 'extra trees', {'max_features': 0.8, 'n_estimators': 700}]

#[0.82016241923201816, 'extra trees', {'max_features': 0.5, 'n_estimators': 500}]
#[0.82030796001624728, 'extra trees', {'max_features': 0.5, 'n_estimators': 700}]
#[0.8202756968581657, 'extra trees', {'max_features': 0.8, 'n_estimators': 500}]
#[0.8210353176616414, 'extra trees', {'max_features': 0.8, 'n_estimators': 700}]
#[0.82046958009256221, 'extra trees', {'max_features': None, 'n_estimators': 500}]
#[0.82097060981518077, 'extra trees', {'max_features': None, 'n_estimators': 700}]
def grid_search(X_train,y,clfs):
    print "grid searching"
    for name,clf in clfs.iteritems(): 
            print name 
            param_grid=clfs[name]['grid']
            param_list = list(ParameterGrid(param_grid))
            for i in range(0,len(param_list)):
                   reg=clfs[name]['est'].set_params(**param_list[i])
                   cv=cv_score1(reg,X_train,y)
                   print [cv.mean(),name,param_list[i]]

def grid_search1(X_train,y,clfs):
    for name, clf in clfs.iteritems(): 
        clf = GridSearchCV(clfs[name]['est'], clfs[name]['grid'], n_jobs=16, verbose=1, cv=5)
        clf.fit(X_train,y)
        print clf.score
        print clf.best_score_
        print clf.best_params_

def main():
    train=load_data('train.csv')
    feature_cols= [col for col in train.columns if col  not in ['target','id']]
    X_train,y_train=split_data(train,feature_cols)
    grid_search(X_train[feature_cols],y_train,get_clfs())

#if __name__ == '__main__':
#    main()
le = preprocessing.LabelEncoder()
data=load_data('train.csv')
train=data.loc[np.random.choice(data.index,np.around(len(data)*0.5), replace=False)]
le.fit(train['target'])
train['target']=le.transform(train['target'])
feature_cols= [col for col in train.columns if col  not in ['target','id']]
X_train,y_train=split_data(train,feature_cols)
grid_search(X_train[feature_cols],y_train,get_extra_trees())
