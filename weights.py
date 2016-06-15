import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import os
from xgb_wrapper import XGBoostClassifier
from sklearn import preprocessing


train = pd.read_csv("train.csv")
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))

le = preprocessing.LabelEncoder()
le.fit(train['target'])
train['target']=le.transform(train['target'])
labels = train['target']
train.drop(['target', 'id'], axis=1, inplace=True)

print(train.head())

### we need a test set that we didn't train on to find the best weights for combining the classifiers
sss = StratifiedShuffleSplit(labels, test_size=0.05, random_state=1234)
for train_index, test_index in sss:
    break

train_x, train_y = train.values[train_index], labels.values[train_index]
test_x, test_y = train.values[test_index], labels.values[test_index]

### building the classifiers
clfs = []

xgb1=XGBoostClassifier(alpha=0, booster='gbtree', colsample_bytree=0.391995799463,
         early_stopping_rounds=30, eta=0.0238249854939,
         eval_metric='mlogloss', gamma=0.0339065215885, l=0, lambda_bias=0,
         max_delta_step=3, max_depth=19, min_child_weight=8, nthread=4,
         ntree_limit=0, num_class=9, num_round=2000,
         objective='multi:softprob', seed=463324, silent=1,
         subsample=0.732463140484, use_buffer=True)

xgb1.fit(train_x, train_y)
print('xgb1 LogLoss {score}'.format(score=log_loss(test_y, xgb1.predict_proba(test_x))))
clfs.append(xgb1)

xgb2=XGBoostClassifier(alpha=0, booster='gbtree', colsample_bytree=0.391930355268,
         early_stopping_rounds=30, eta=0.0310539855081,
         eval_metric='mlogloss', gamma=0.0805700186495, l=0, lambda_bias=0,
         max_delta_step=7, max_depth=10, min_child_weight=5, nthread=4,
         ntree_limit=0, num_class=9, num_round=2000,
         objective='multi:softprob', seed=879261, silent=1,
         subsample=0.512480226074, use_buffer=True)
xgb2.fit(train_x, train_y)
print('xgb2 LogLoss {score}'.format(score=log_loss(test_y, xgb2.predict_proba(test_x))))
clfs.append(xgb2)

xgb3=XGBoostClassifier(alpha=0, booster='gbtree', colsample_bytree=0.793904376243,
         early_stopping_rounds=30, eta=0.021569491826,
         eval_metric='mlogloss', gamma=0.0673376079738, l=0, lambda_bias=0,
         max_delta_step=7, max_depth=20, min_child_weight=6, nthread=4,
         ntree_limit=0, num_class=9, num_round=1000,
         objective='multi:softprob', seed=808366, silent=0,
         subsample=0.924346706155, use_buffer=True)

xgb3.fit(train_x, train_y)
print('xgb3 LogLoss {score}'.format(score=log_loss(test_y, xgb3.predict_proba(test_x))))
clfs.append(xgb3)

xgb4=XGBoostClassifier(alpha=0, booster='gbtree', colsample_bytree=0.427739942719,
         early_stopping_rounds=30, eta=0.0273684221922,
         eval_metric='mlogloss', gamma=0.0765202451393, l=0, lambda_bias=0,
         max_delta_step=10, max_depth=11, min_child_weight=4, nthread=4,
         ntree_limit=0, num_class=9, num_round=1000,
         objective='multi:softprob', seed=632569, silent=0,
         subsample=0.728260256802, use_buffer=True)
xgb4.fit(train_x, train_y)
print('xgb4 LogLoss {score}'.format(score=log_loss(test_y, xgb4.predict_proba(test_x))))
clfs.append(xgb4)

xgb5=XGBoostClassifier(alpha=0, booster='gbtree', colsample_bytree=0.4709696894,
         early_stopping_rounds=30, eta=0.0288268870077,
         eval_metric='mlogloss', gamma=0.0465269290682, l=0, lambda_bias=0,
         max_delta_step=10, max_depth=20, min_child_weight=6, nthread=4,
         ntree_limit=0, num_class=9, num_round=1000,
         objective='multi:softprob', seed=335744, silent=0,
         subsample=0.700744685247, use_buffer=True)
xgb5.fit(train_x, train_y)
print('xgb5 LogLoss {score}'.format(score=log_loss(test_y, xgb5.predict_proba(test_x))))
clfs.append(xgb5)

xgb6=XGBoostClassifier(alpha=0, booster='gbtree', colsample_bytree=0.539822278005,
         early_stopping_rounds=30, eta=0.0544945482968,
         eval_metric='mlogloss', gamma=0.0577087363402, l=0, lambda_bias=0,
         max_delta_step=7, max_depth=11, min_child_weight=6, nthread=4,
         ntree_limit=0, num_class=9, num_round=1000,
         objective='multi:softprob', seed=854809, silent=0,
         subsample=0.940582910512, use_buffer=True)
xgb6.fit(train_x, train_y)
print('xgb6 LogLoss {score}'.format(score=log_loss(test_y, xgb6.predict_proba(test_x))))
clfs.append(xgb6)

xgb7=XGBoostClassifier(alpha=0, booster='gbtree', colsample_bytree=0.459971793632,
         early_stopping_rounds=30, eta=0.0305648288294,
         eval_metric='mlogloss', gamma=0.0669039612464, l=0, lambda_bias=0,
         max_delta_step=4, max_depth=14, min_child_weight=8, nthread=4,
         ntree_limit=0, num_class=9, num_round=1000,
         objective='multi:softprob', seed=84425, silent=0,
         subsample=0.972607582489, use_buffer=True)
xgb7.fit(train_x, train_y)
print('xgb7 LogLoss {score}'.format(score=log_loss(test_y, xgb7.predict_proba(test_x))))
clfs.append(xgb7)


### finding the optimum weights

predictions = []
for clf in clfs:
    predictions.append(clf.predict_proba(test_x))

def log_loss_func(weights):
    ''' scipy minimize will pass the weights as a numpy array '''
    final_prediction = 0
    for weight, prediction in zip(weights, predictions):
            final_prediction += weight*prediction

    return log_loss(test_y, final_prediction)
    
#the algorithms need a starting value, right not we chose 0.5 for all weights
#its better to choose many random starting points and run minimize a few times
starting_values = [0.5]*len(predictions)

#adding constraints  and a different solver as suggested by user 16universe
cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
#our weights are bound between 0 and 1
bounds = [(0,1)]*len(predictions)

res = minimize(log_loss_func, starting_values, method='SLSQP', bounds=bounds, constraints=cons)

print('Ensamble Score: {best_score}'.format(best_score=res['fun']))
print('Best Weights: {weights}'.format(weights=res['x']))