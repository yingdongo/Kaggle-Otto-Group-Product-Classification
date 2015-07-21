# -*- coding: utf-8 -*-
"""
Created on Sun May  3 22:03:34 2015

@author: Ying
"""

from tools import load_data
from sklearn import ensemble
from feature_engineering import feature_engineering
import pandas as pd
from tools import cv_score
from sklearn import preprocessing
from tools import cross_v
from sklearn.calibration import CalibratedClassifierCV
from sklearn import cross_validation
import xgboost as xgb
from xgb_wrapper import XGBoostClassifier

def write_submission(test_ids,preds,filename):
    preds = pd.DataFrame(preds, columns=['Class_1', 'Class_2', 'Class_3',
                                             'Class_4', 'Class_5', 'Class_6',
                                             'Class_7', 'Class_8', 'Class_9'])
    
    submission = pd.concat([test_ids, preds], axis=1)
    file_name = filename
    submission.to_csv(file_name, index=False)
    
def get_rf():
     return ensemble.RandomForestClassifier(n_estimators=100)
     
def get_gb():
     return ensemble.GradientBoostingClassifier(n_estimators=100)
def get_tuned_gb():
     return ensemble.GradientBoostingClassifier(n_estimators=500)
def get_tuned_rf():
     return ensemble.RandomForestClassifier(n_estimators=800,max_features=0.6)

def benchmark_solution():
    train=load_data('train.csv')
    test=load_data('test.csv')
    lbl_enc = preprocessing.LabelEncoder()
    train['target'] = lbl_enc.fit_transform(train['target'])
    feature_cols= [col for col in train.columns if col  not in ['target','id']]
    X_train=train[feature_cols]
    y=train['target']
    X_test=test[feature_cols]
    test_ids=test['id']
    print "benchmark solution"
    cross_v(get_rf(),X_train.values,y.values)#0.596256539386
    #clf=get_rf()
    #clf.fit(X_train,y)
    #preds = clf.predict_proba(X_test)
    #write_submission(test_ids,preds,'submissions/benchmark_solutionrf100.csv')
    #0.58653

def feature_engineering_solution():
    train=load_data('train.csv')
    test=load_data('test.csv')
    le = preprocessing.LabelEncoder()
    le.fit(train['target'])
    train['target']=le.transform(train['target'])
    feature_cols= [col for col in train.columns if col  not in ['target','id']]
    X_train=feature_engineering(train[feature_cols])
    X_test=feature_engineering(test[feature_cols])
    feature_cols= [col for col in X_train.columns]#std 0.607958003167 mean 0.615741311533
    X_train=X_train[feature_cols]
    X_test=X_test[feature_cols]
    y=train['target']
    test_ids=test['id']
    print 'feature_engineering_solution'
    cross_v(get_rf(),X_train.values,y.values)#0.600017926514
    #clf=get_rf()
    #clf.fit(X_train,y)
    #preds = clf.predict_proba(X_test)
    #write_submission(test_ids,preds,'submissions/feature_engineering_rf100.csv')
    #0.59432

def feature_selection_solution():
    train=load_data('train.csv')
    test=load_data('test.csv')
    le = preprocessing.LabelEncoder()
    le.fit(train['target'])
    train['target']=le.transform(train['target'])
    feature_cols= [col for col in train.columns if col  not in ['target','id']]
    X_train=feature_engineering(train[feature_cols])
    X_test=feature_engineering(test[feature_cols])
    feature_cols=[col for col in X_train.columns if col not in ['mean','std','nonzero','feat_6','feat_82','feat_84']]
    X_train=X_train[feature_cols]
    X_test=X_test[feature_cols]
    print X_train.columns
    y=train['target']
    test_ids=test['id']
    print 'feature_selection_solution'
    cross_v(get_rf(),X_train.values,y.values)# mean 0.595288515439   std 0.593551044059 nonzero  0.597406303207
    #no fg 6 82 84 0.603600594376
    #0.600058535601
    clf=get_rf()
    clf.fit(X_train,y)
    preds = clf.predict_proba(X_test)
    write_submission(test_ids,preds,'submissions/feature_selection_rf100_84_82_6_nofg.csv')
    # no mean 6 82 84 0.58840
    # no engineering no 6 82 84 0.58961

    
def model_selection_solution():
    train=load_data('train.csv')
    test=load_data('test.csv')
    le = preprocessing.LabelEncoder()
    le.fit(train['target'])
    train['target']=le.transform(train['target'])
    feature_cols= [col for col in train.columns if col  not in ['target','id']]
    X_train=train[feature_cols]
    X_test=test[feature_cols]
    y=train['target']
    test_ids=test['id']
    print 'gbrt_tuned_selection_solution'
    #cross_v(get_tuned_gb(),X_train.values,y.values)
    clf=get_tuned_gb()
    clf.fit(X_train,y)
    preds = clf.predict_proba(X_test)
    write_submission(test_ids,preds,'submissions/gbrt_tuned_selection_solution.csv')
    # cv 0.60090192958 leader bord 0.59601
    
def parameter_tuning_solution():
    train=load_data('train.csv')
    test=load_data('test.csv')
    le = preprocessing.LabelEncoder()
    le.fit(train['target'])
    train['target']=le.transform(train['target'])
    feature_cols= [col for col in train.columns if col  not in ['target','id']]
    X_train=train[feature_cols]
    X_test=test[feature_cols]
    y=train['target']
    test_ids=test['id']
    print 'parameter_tuning_solution800_6'
    cross_v(get_tuned_rf(),X_train.values,y.values)#0.546637992781
    clf=get_tuned_rf()
    clf.fit(X_train,y)
    preds = clf.predict_proba(X_test)
    write_submission(test_ids,preds,'submissions/parameter_tuning_solution800_6.csv')
    # 0.53773_600_0.5  0.53472_800 0.6
    
def rf_calibration_solution():
    train=load_data('train.csv')
    test=load_data('test.csv')
    le = preprocessing.LabelEncoder()
    le.fit(train['target'])
    train['target']=le.transform(train['target'])
    feature_cols= [col for col in train.columns if col  not in ['target','id']]
    X_train=train[feature_cols]
    X_test=test[feature_cols]
    y=train['target']
    test_ids=test['id']
    print 'rf_calibration_solution'
    skf = cross_validation.StratifiedKFold(y, n_folds=5, random_state=42)
    calibration_method = 'isotonic'
    clf=get_tuned_rf()
    ccv = CalibratedClassifierCV(base_estimator=clf, method=calibration_method, cv=skf)
    ccv.fit(X_train,y)
    preds = ccv.predict_proba(X_test)
    write_submission(test_ids,preds,'submissions/rf_calibration_solution.csv')
    #0.48998 5cv 800 0.5 rf 0.49390 rf 5cv 1000 0.6

def xgboost_solution():
    train=load_data('train.csv')
    test=load_data('test.csv')
    le = preprocessing.LabelEncoder()
    le.fit(train['target'])
    train['target']=le.transform(train['target'])
    feature_cols= [col for col in train.columns if col  not in ['target','id']]
    X_train=train[feature_cols]
    X_test=test[feature_cols]
    y=train['target']
    test_ids=test['id']
    print 'rf_calibration_solution'
    dtrain = xgb.DMatrix(X_train, label=y)
    dtest = xgb.DMatrix(X_test, label=None)
    param = {'bst:max_depth':10, 'bst:min_child_weight': 4, 'bst:subsample': 0.5, 'bst:colsample_bytree':0.8,  'bst:eta':0.05}
    other = {'silent':1, 'objective':'multi:softprob', 'num_class':9, 'nthread': 4, 'eval_metric': 'mlogloss', 'seed':0}
    watchlist  = [(dtrain,'train')]
    full_param = other.copy()
    full_param.update(param)
    plst = full_param.items()
    bst= xgb.train(plst, dtrain, 300, watchlist)
    preds = bst.predict(dtest)
    write_submission(test_ids,preds,'submissions/xgboost_solution.csv')
    #default parameters 0.47409
    
def xgboost_param_solution():
    xgb=XGBoostClassifier(alpha=0, booster='gbtree', colsample_bytree=0.459971793632,
         early_stopping_rounds=30, eta=0.0305648288294,
         eval_metric='mlogloss', gamma=0.0669039612464, l=0, lambda_bias=0,
         max_delta_step=4, max_depth=14, min_child_weight=8, nthread=4,
         ntree_limit=0, num_class=9, num_round=1000,
         objective='multi:softprob', seed=84425, silent=0,
         subsample=0.972607582489, use_buffer=True)

    train=load_data('train.csv')
    test=load_data('test.csv')
    le = preprocessing.LabelEncoder()
    le.fit(train['target'])
    train['target']=le.transform(train['target'])
    feature_cols= [col for col in train.columns if col  not in ['target','id']]
    X_train=train[feature_cols]
    X_test=test[feature_cols]
   
    y=train['target']
    test_ids=test['id']
    
    xgb.fit(X_train, y)
    preds=xgb.predict_proba(X_test)
    write_submission(test_ids,preds,'submissions/xgboost_param_solution_76.csv')
#65 0.43237

    
def main():
    #benchmark_solution()
    #feature_engineering_solution()
    #feature_selection_solution()
    #parameter_tuning_solution()
    #rf_calibration_solution()
    #xgboost_solution()
    model_selection_solution()
if __name__ == '__main__':
    main()