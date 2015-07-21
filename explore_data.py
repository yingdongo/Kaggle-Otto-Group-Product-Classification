# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 11:48:10 2015

@author: Ying
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 17:23:41 2015

@author: Ying
"""
from tools import load_data
import numpy as np
import matplotlib.pyplot as plt
import pylab as pb
from sklearn import preprocessing
import pandas as pd
plt.style.use('ggplot')
pb.style.use('ggplot')
def show_data(data):
    print data.head(10)
    #have a look at few top rows
    print data.describe()
    #describe() function would provide count, mean, standard deviation (std), min, quartiles and max in its output

def plot_cor(data):
    """Plot pairwise correlations of features in the given dataset"""
    from matplotlib import cm

    cols = data.columns.tolist()
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    
    # Plot absolute value of pairwise correlations since we don't
    # particularly care about the direction of the relationship,
    # just the strength of it
    cax = ax.matshow(data.corr().abs(), cmap=cm.YlOrRd)
    
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(cols)))
    ax.set_xticklabels(cols)
    ax.set_yticklabels(cols)
    ax.set_title("Correlation Matrix of Features")

#def main():   
    train=load_data('train.csv')
    cols = [col for col in train.columns if col not in ['id','target']] 
    features=['feat_34','feat_11','feat_40','feat_26','feat_60','feat_25','feat_86','feat_15','feat_90','feat_14','feat_42','feat_67','feat_62','feat_36','feat_24','target']
    #xCol = 'target'
    #for col in cols:
    #    plotHist(train, col,xCol)
    x_data=train[features]
    #plot_cor(x_data)
    y=train['target']
    class1=y[[y=='class1']]
#if __name__ == '__main__':
#    main()

train=load_data('train.csv')
cols = [col for col in train.columns if col not in ['id','target']] 
features=['feat_34','feat_11','feat_40','feat_26','feat_60','feat_25','feat_86','feat_15','feat_90','feat_14','feat_42','feat_67','feat_62','feat_36','feat_24','target']
#xCol = 'target'
#for col in cols:
#    plotHist(train, col,xCol)
x_data=train[features]
#plot_cor(x_data)
y=train['target']
class1=y[y=='Class_1'].count()
class2=y[y=='Class_2'].count()
class3=y[y=='Class_3'].count()
class4=y[y=='Class_4'].count()
class5=y[y=='Class_5'].count()
class6=y[y=='Class_6'].count()
class7=y[y=='Class_7'].count()
class8=y[y=='Class_8'].count()
class9=y[y=='Class_9'].count()
values=np.array([class1,class2,class3,class4,class5,class6,class7,class8,class9])
preds = pd.DataFrame(values, index=['Class_1', 'Class_2', 'Class_3',
                                             'Class_4', 'Class_5', 'Class_6',
                                             'Class_7', 'Class_8', 'Class_9'])
labels=['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6','Class_7', 'Class_8', 'Class_9']


import numpy as np
import matplotlib.pyplot as plt

N = 9
menMeans = values

ind = np.arange(N)  # the x locations for the groups
width = 0.7      # the width of the bars

fig, ax = plt.subplots(figsize=(10,7))
rects1 = ax.bar(ind, menMeans, width)

# add some text for labels, title and axes ticks
ax.set_title('Distribution of Product Categories')
ax.set_xticks(ind+width)
ax.set_xticklabels(labels)
for tl in ax.get_xticklabels():
    tl.set_fontsize(13)
for tl in ax.get_yticklabels():
    tl.set_fontsize(13)
    
def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%d'%int(height),
                ha='center', va='bottom')

autolabel(rects1)

plt.show()


