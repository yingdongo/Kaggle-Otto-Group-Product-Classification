# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 00:28:47 2015

@author: Ying
"""
from sklearn.grid_search import GridSearchCV
from sklearn import ensemble
from feature_engineering import feature_engineering
from tools import load_data
from tools import cross_v
from sklearn.grid_search import ParameterGrid
import xgboost as xgb
from sklearn import preprocessing

def get_clfs():
    return {
            'extra trees' : { 
                'est' :ensemble.ExtraTreesClassifier(),
                'grid' : {
                    'n_estimators' : [100,500],
                    'max_features': [.1, .5, None ]
                }
            },
#0.820776366398
#{'max_features': 0.5, 'n_estimators': 500}
            'gbrt' : { 
                'est' :ensemble.GradientBoostingClassifier(),
                'grid' : {
                    'n_estimators' : [100,500],
                    'learning_rate': [0.1,0.03,0.01],
                }
            },
#0.804857946281
#{'n_estimators': 500, 'learning_rate': 0.1}
            'random forests' : { 
                'est' :ensemble.RandomForestClassifier(),
                'grid' : {
                    'n_estimators' : [100,500],
                    'max_features': [ .1, .5, None ],
                }
            },
#0.811322279324
#{'max_features': 0.1, 'n_estimators': 500}
        }
def get_gb():
    return {
            'gb' : { 
                'est' :ensemble.GradientBoostingClassifier(),
                'grid' : {
                    'n_estimators' : [100,500],
                    'learning_rate': [0.1,0.03,0.01],

                }
            }
        }#{'n_estimators': 500, 'learning_rate': 0.1}
         #Results: 0.520820174717
def get_gb():
    return {
            'gb' : { 
                'est' :ensemble.GradientBoostingClassifier(),
                'grid' : {
                    'n_estimators' : [100,500],
                    'learning_rate': [0.1,0.03,0.01],

                }
            }
        }#{'n_estimators': 500, 'learning_rate': 0.1}
         #Results: 0.520820174717
def get_rf():
    return {
            'random forests' : { 
                'est' :ensemble.RandomForestClassifier(),
                'grid' : {
                    'n_estimators' : [600,800,1000],
                    'max_features': [.5,.6,.7]
                }
            }
        }
#{'max_features': 0.6, 'n_estimators': 1000}
#Results: 0.546649262304

def grid_search(X_train,y,clfs):
    for name, clf in clfs.iteritems(): 
        params = clfs[name]['grid']
        clf=clfs[name]['est'](**params)
        print name
        #cross_v(clf,X_train.values,y.values)
        clf.fit(X_train.values,y.values)
        
def main():
    train=load_data('train.csv')
    feature_cols= [col for col in train.columns if col  not in ['target','id']]
    X_train=feature_engineering(train[feature_cols])
    y=train['target']
    grid_search(X_train,y,get_clfs())

#if __name__ == '__main__':
#    main()
train=load_data('train.csv')
le = preprocessing.LabelEncoder()
le.fit(train['target'])
train['target']=le.transform(train['target'])
feature_cols= [col for col in train.columns if col  not in ['target','id']]
X_train=train[feature_cols]
y=train['target']
clfs=get_gb()
for name, clf in clfs.iteritems(): 
    print name
    param_list = list(ParameterGrid(clfs[name]['grid']))
    for i in range(0,len(param_list)):
        clf=clfs[name]['est'].set_params(**param_list[i])
        print clf
        print param_list[i]
        cross_v(clf,X_train.values,y.values)
