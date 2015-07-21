# -*- coding: utf-8 -*-
"""
Created on Sat May 16 23:35:17 2015

@author: Ying
"""
import pandas as pd
pred1=pd.read_csv('submissions/xgboost_param_solution_14.csv')
pred2=pd.read_csv('submissions/xgboost_param_solution_26.csv')
pred4=pd.read_csv('submissions/xgboost_param_solution_50.csv')
pred5=pd.read_csv('submissions/xgboost_param_solution_59.csv')
pred6=pd.read_csv('submissions/xgboost_param_solution_65.csv')
pred7=pd.read_csv('submissions/xgboost_param_solution_68.csv')
pred8=pd.read_csv('submissions/xgboost_param_solution_76.csv')

#preds=(pred1+pred2+pred4+pred5+pred6+pred7+pred8)/7.0
preds=pred1.values*0.14286186+pred2*0.14286467+pred4*0.14286459+pred5*0.14286459+pred6*0.14283767+pred7*0.14287165+pred8*0.14285376

preds = preds.drop(['id'], axis=1)
preds.index+=1

headers = ["Class_1","Class_2","Class_3","Class_4","Class_5","Class_6","Class_7","Class_8","Class_9"]

preds.to_csv("submissions/xgboost_7model_stacking_weights.csv", header=headers,index=True,index_label = 'id')

#8 0.42936
#4 0.42941


#==============================================================================
# pred1=pd.read_csv('submissions/xgboost_param_solution_14.csv')
# pred2=pd.read_csv('submissions/xgboost_param_solution_26.csv')
# pred4=pd.read_csv('submissions/xgboost_param_solution_50.csv')
# pred5=pd.read_csv('submissions/xgboost_param_solution_59.csv')
# pred6=pd.read_csv('submissions/xgboost_param_solution_65.csv')
# pred7=pd.read_csv('submissions/xgboost_param_solution_68.csv')
# pred8=pd.read_csv('submissions/xgboost_param_solution_76.csv')
# 
# pred1=pred1.drop(['id'], axis=1)
# pred2=pred2.drop(['id'], axis=1)
# pred4=pred4.drop(['id'], axis=1)
# pred5=pred5.drop(['id'], axis=1)
# pred6=pred6.drop(['id'], axis=1)
# pred7=pred7.drop(['id'], axis=1)
# pred8=pred8.drop(['id'], axis=1)
# 
# ### finding the optimum weights
# 
# predictions = []
# predictions.append(pred1.values)
# predictions.append(pred2.values)
# predictions.append(pred4.values)
# predictions.append(pred5.values)
# predictions.append(pred6.values)
# predictions.append(pred7.values)
#==============================================================================
#predictions.append(pred8.values)
