import pandas as pd
import numpy as np
from xgb_wrapper import XGBoostClassifier
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import random
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import log_loss
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing

random.seed()

# Returns train_x,valid_x in 2d, train_y, valid_y in 1d  and start with '0'
def loadData():
    # Preprocess

    features=[]
    for i in range(1,94):
        features.append("feat_"+str(i))

    # Load data
    train_data = pd.read_csv("train.csv")
    le = preprocessing.LabelEncoder()
    le.fit(train_data['target'])
    train_data['target']=le.transform(train_data['target'])
    
    train, valid,train_y, valid_y = train_test_split(train_data[features], train_data['target'], test_size=0.1, random_state=42)

    test = pd.read_csv("test.csv")
    
  
    train_x=train[features].values
    valid_x=valid[features].values
    train_y=train_y.values
    valid_y=valid_y.values
    test_x=test[features].values
    

    train_x = train_x.astype(np.float64)
    valid_x = valid_x.astype(np.float64)
    train_y = train_y.astype(np.float64)
    valid_y = valid_y.astype(np.float64)
    test_x = test_x.astype(np.float64)
    print('Data has been loaded!')
    return train_x,train_y,valid_x,valid_y,test_x

# Saves predictions(2-d numpy array) into './results/results.csv'
def saveData(predictions,fpath):
    df = pd.DataFrame(predictions) #predictions is a numpy 2d array
    df.index+=1
    headers = ["Class_1","Class_2","Class_3","Class_4","Class_5","Class_6","Class_7","Class_8","Class_9"]
    df.to_csv(fpath, header=headers,index=True, index_label = 'id')
    print('Predictions has been saved!')

train_x,train_y,valid_x,valid_y,test_x=loadData()
# normalization
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
valid_x = scaler.transform(valid_x)
test_x = scaler.transform(test_x)

def trainrf(model_id,train_x,train_y,valid_x,valid_y,test_x):
    train_x,train_y=shuffle(train_x,train_y)


    random_state=random.randint(0, 1000000)
    print('random state: {state}'.format(state=random_state))

    clf = RandomForestClassifier(n_estimators=random.randint(50,5000),
                                 criterion='gini',
                                 max_depth=random.randint(10,1000),
                                 min_samples_split=random.randint(2,50),
                                 min_samples_leaf=random.randint(1,10),
                                 min_weight_fraction_leaf=random.uniform(0.0,0.5),
                                 max_features=random.uniform(0.1,1.0),
                                 max_leaf_nodes=random.randint(1,10),
                                 bootstrap=False,
                                 oob_score=False,
                                 n_jobs=30,
                                 random_state=random_state,
                                 verbose=0,
                                 warm_start=True,
                                 class_weight=None
                )

    clf.fit(train_x, train_y)

    valid_predictions1 = clf.predict_proba(valid_x)
    test_predictions1= clf.predict_proba(test_x)

    t1 = test(valid_y,valid_predictions1)

    ccv = CalibratedClassifierCV(base_estimator=clf,method="sigmoid",cv='prefit')
    ccv.fit(valid_x,valid_y)

    valid_predictions2 = ccv.predict_proba(valid_x)
    test_predictions2= ccv.predict_proba(test_x)

    t2 = test(valid_y,valid_predictions2)

    if t2<t1:
        valid_predictions=valid_predictions2
        test_predictions=test_predictions2
        t=t2
    else:
        valid_predictions=valid_predictions1
        test_predictions=test_predictions1
        t=t1

    if t < 0.450:
        saveData(valid_predictions,"../valid_results/valid_"+str(model_id)+".csv")
        saveData(test_predictions,"../results/results_"+str(model_id)+".csv")


def test(valid_y,valid_predictions):
    class2num={}
    for label in range(9):
        tmp=[0]*9
        for i in range(9):
            if i == label:
                tmp[i]=1.0
        class2num[label]=tmp

    _valid_y=[]
    for i in range(0,len(valid_y)):
        _valid_y.append(class2num[valid_y[i]])
    _valid_y=np.array(_valid_y)

    print('valid loss:')
    print(log_loss(valid_y,valid_predictions))

def trainxgb(model_id,train_x,train_y,valid_x,valid_y,test_x):
    train_x,train_y=shuffle(train_x,train_y)

    random_state=random.randint(0, 1000000)
    print('random state: {state}'.format(state=random_state))

    xgboost = XGBoostClassifier(silent=0,
                 objective='multi:softprob',
                 eval_metric='mlogloss',
                 num_class=9,
                 nthread=4,
                 seed=random_state,
                 eta=random.uniform(0.01,0.1),
                 max_depth=random.randint(10,20),
                 max_delta_step=random.randint(1,10),
                 min_child_weight=random.randint(1,10),
                 subsample=random.uniform(0.0,1.0),
                 gamma=random.uniform(0.01,0.1),
                 colsample_bytree=random.uniform(0.0,1.0),
                 early_stopping_rounds=30,
                 num_round=1000
                )
      
    xgboost.fit(train_x, train_y)
    
    valid_predictions = xgboost.predict_proba(valid_x)
    score =log_loss(valid_y,valid_predictions)
    print score
    fName = open("report_xgb_"+str(model_id)+".txt", 'w')
    print >> fName, "score:"
    print >> fName, score
    print >>fName, "model_id:"
    print >>fName, model_id
    print >> fName, xgboost
    fName.close()
    #if  score<0.450:
    #    test_predictions= xgboost.predict_proba(test_x)
    #    saveData(valid_predictions,"submissions/valid_"+str(model_id)+".csv")
    #    saveData(test_predictions,"submissions/results_"+str(model_id)+str(score)+".csv")

for model_id in range(46,100):
       print('model_id: {state}'.format(state=model_id))
       trainxgb(model_id,train_x,train_y,valid_x,valid_y,test_x)


  