# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 18:40:17 2015

@author: Ying
"""

from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import ensemble
from tools import cross_v
from tools import split_data
import pandas as pd
from matplotlib import pyplot as plt
from tools import load_data
from sklearn import preprocessing

def create_clf():
    models=[]
    models.append(('',))
    #models.append(('KNeighborsClassifier',KNeighborsClassifier()))
    #models.append(('AdaBooost',ensemble.AdaBoostClassifier()))
    #models.append(('ExtraTrees',ensemble.ExtraTreesClassifier(n_estimators=100)))
    #models.append(('GB',ensemble.GradientBoostingClassifier(n_estimators=100)))#0.600893
    #models.append(('RandomForest',ensemble.RandomForestClassifier(n_estimators=100)))
    return models
#KNeighborsClassifier 2.394348
#ExtraTrees    0.610466
#GB            0.600852
#RandomForest  0.595299
def clf_score(models,X_train,y_train):
    index=[]
    score=[]
    for clf in models:
        index.append(clf[0])
        cv=cross_v(clf[1],X_train.values,y_train.values)
        print clf[0]
        print cv
        score.append(cv)
    return pd.DataFrame(score,index=index)
    
def main():
    train=load_data('train.csv')
    lbl_enc = preprocessing.LabelEncoder()
    train['target'] = lbl_enc.fit_transform(train['target'])
    feature_cols= [col for col in train.columns if col  not in ['target','id']]
    X_train,y_train=split_data(train,feature_cols)
    clf_scores=clf_score(create_clf(),X_train[feature_cols],y_train)
    print clf_scores
    plt.plot(clf_scores)
    plt.xticks(range(len(clf_scores)), clf_scores.index, fontsize=14, rotation=90)
    plt.show()

if __name__ == '__main__':
    main()
                     