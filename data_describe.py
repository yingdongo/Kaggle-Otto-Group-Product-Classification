# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 00:07:27 2015

@author: Ying
"""
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

pred=pd.read_csv('submissions/parameter_tuning_solution800_6.csv')
pred1=pd.read_csv('submissions/rf_calibration_solution.csv')

pred=pred.drop(['id'],axis=1)
pred1=pred1.drop(['id'],axis=1)

plt.figure(figsize=(12,10))
plt.hist([pred.values.flatten(),pred1.values.flatten()],label=['Random Forest','calibrated Random Forest'])
plt.legend(fontsize=14)
plt.xlabel('Predicted probability',fontsize=16)
plt.ylabel('Frequency',fontsize=16)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
