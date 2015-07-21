# -*- coding: utf-8 -*-
"""
Created on Mon May  4 17:12:28 2015

@author: Ying
"""
from sklearn import ensemble
from tools import load_data
from sklearn.metrics import log_loss
from sklearn.calibration import CalibratedClassifierCV
from feature_engineering import feature_engineering
from sklearn import cross_validation
from tools import cross_v
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def get_rf():
    forest=ensemble.RandomForestClassifier(n_estimators=100)
    return forest
    
train=load_data('train.csv')
feature_cols = [col for col in train.columns if col not in  ['id','target']] 
X_train=feature_engineering(train[feature_cols]).values
y=train['target'].values
X_train, X_test, y_train, y_test=cross_validation.train_test_split(X_train,y,test_size=0.33,random_state=1)

skf = cross_validation.StratifiedKFold(y_train, n_folds=10, random_state=42)
calibration_method = 'isotonic'
clf=get_rf()
ccv = CalibratedClassifierCV(base_estimator=clf, method=calibration_method, cv=skf)

#ccv.fit(X_train,y_train)
#pred=ccv.predict_proba(X_test)
clf.fit(X_train,y_train)
pred=clf.predict_proba(X_test)
score=log_loss(y_test,pred)
#0.487707826761

plt.hist(pred)