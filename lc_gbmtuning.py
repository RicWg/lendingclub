# dev experiment log
# v0.2 GBM paramter tuning
# lgb.LGBMClassifier(n_estimators=500)
# perf: final:  [{<function modelGBM at 0x10bcb6048>: 0.4471998532954069},
# {<function modelGBM at 0x10bcb6048>: 0.44832123696627835},
# {<function modelGBM at 0x10bcb6048>: 0.4473461706562614}]
# est= 500, max_bin = 100
# final:  [{<function modelGBM at 0x10c439048>: 0.4473790039046676},
# {<function modelGBM at 0x10c439048>: 0.44895739202204954},
# {<function modelGBM at 0x10c439048>: 0.4475065456074244}]
# est = 500, learning_rate = 0.05
# perf: final:  [{<function modelGBM at 0x109b04048>: 0.44558207230986013},
# {<function modelGBM at 0x109b04048>: 0.4469639912169554},
# {<function modelGBM at 0x109b04048>: 0.44598296350893973}]
# est=500, lr=0.03, no improvement
# num_leaves to 200
#final:  [{<function modelGBM at 0x10e0b1048>: 0.44558069583294774},
# {<function modelGBM at 0x10e0b1048>: 0.44686052491033906},
# {<function modelGBM at 0x10e0b1048>: 0.4458362713878767}]


import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn import tree
from sklearn import ensemble
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

import time
start_time = time.time()
print("Start Project LC...")

def devprep():
    data = pd.read_csv('loan_stat542.csv')
    print(data.head())
    print("loan_stat542.csv loaded! shape : ", data.shape)
    # split into 3 train/test.csv
    testid = pd.read_csv('Project3_test_id.csv')

    for fold in range(3):
        traindata = data[-data['id'].isin(testid['test' + str(fold + 1)].astype(int))]
        testdata = data[data['id'].isin(testid['test' + str(fold + 1)].astype(int))]
        path = "./" + str(fold+1)
        os.mkdir(path)
        traindata.to_csv(path +"/train.csv", index=False)
        testdata.to_csv(path + "/test.csv", index=False)
    print("Train/test data split done!")


def preproc0(traindata, testdata):
    trainsize = len(traindata)
    dt = pd.concat(objs=[traindata, testdata], axis=0, sort=True)

    # recode ['Charged Off', 'Default'] to 1, ['Fully Paid'] =0
    recoMap = {'loan_status': {'Fully Paid': 0, 'Default': 1, 'Charged Off': 1}}
    for f in recoMap:
        print(f)
        dt.replace(recoMap[f], inplace=True)

    print("Column with NA: ", dt.columns[dt.isna().any()].tolist() )
    dt['emp_length'].fillna(dt['emp_length'].mode()[0], inplace=True)
    dt['dti'].fillna(dt['dti'].mean(), inplace=True)
    dt['revol_util'].fillna(dt['revol_util'].mean(), inplace=True)
    dt['mort_acc'].fillna(dt['mort_acc'].mean(), inplace=True)
    dt['pub_rec_bankruptcies'].fillna(dt['pub_rec_bankruptcies'].mean(), inplace=True)
    print("Column with NA after filling: ", dt.columns[dt.isna().any()].tolist())

    recoMap = {'grade': {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6},
               'term': {'60 months': 0, '36 months': 1},
               'sub_grade': {'A1': 0, 'A2': 1, 'A3': 2, 'A4': 3, 'A5': 4,
                             'B1': 5, 'B2': 6, 'B3': 7, 'B4': 8, 'B5': 9,
                             'C1': 10, 'C2': 11, 'C3': 12, 'C4': 13, 'C5': 14,
                             'D1': 15, 'D2': 16, 'D3': 17, 'D4': 18, 'D5': 19,
                             'E1': 20, 'E2': 21, 'E3': 22, 'E4': 23, 'E5': 24,
                             'F1': 25, 'F2': 26, 'F3': 27, 'F4': 28, 'F5': 29,
                             'G1': 30, 'G2': 31, 'G3': 32, 'G4': 33, 'G5': 34},
               'emp_length': {'< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3,
                              '4 years': 4, '5 years': 5, '6 years': 6, '7 years': 7,
                              '8 years': 8, '9 years': 9, '10+ years': 10},
               'verification_status': {'Not Verified': 0, 'Source Verified': 1, 'Verified': 1}
               }

    for f in recoMap:
        #print(f)
        dt.replace(recoMap[f], inplace=True)

    dt['earliest_cr_line_year'] = dt['earliest_cr_line'].apply(
        lambda x: time.strptime(x, '%b-%Y').tm_year)
    dt['earliest_cr_line_month'] = dt['earliest_cr_line'].apply(
        lambda x: time.strptime(x, '%b-%Y').tm_mon )

    dt = dt.drop(columns=['title', 'zip_code', 'emp_title', 'earliest_cr_line'])
    print("dropping features : ", dt.shape)

    dt = pd.get_dummies(dt)
    print("dummify features : ", dt.shape)

    traindata = dt[:trainsize]
    testdata = dt[trainsize:]

    print("preprocessing done!")
    return traindata, testdata

def eval(trainfea, trainlabel, testfea, testlabel, model):
    print('Training model: ', model)
    model.fit(trainfea, trainlabel)
    train_result = model.predict(trainfea)
    loss = metrics.log_loss(trainlabel, train_result)
    print('Training loss: ', loss)

    test_result = model.predict_proba(testfea)
    loss_test = metrics.log_loss(testlabel, test_result)
    print('Test loss: ', loss_test)
    return loss_test


# models
def modelRF(trainfea, trainlabel, testfea, testlabel, cv=False):
    print("Model RandomForest...")
    if cv:
        md = RandomForestClassifier(n_jobs=-1)
        grid_search = GridSearchCV(
            estimator=md, scoring='neg_log_loss',
            param_grid={
                'n_estimators': [50, 200, 400]#, #[50, 100, 200, 300, 400],
                #'min_samples_leaf': [1, 5, 10],
                #'min_samples_split': [2, 5, 10]

            },
            cv=5)
        grid_search.fit(trainfea, trainlabel)
        print("best parameters: ", grid_search.best_estimator_)
        trainerror = -grid_search.score(trainfea, trainlabel)
        print("best CV error on training set: ", trainerror)

        testpred = grid_search.predict(testfea)
        logloss = metrics.log_loss(testlabel, testpred)
    else:
        md = RandomForestClassifier(n_estimators=50, n_jobs=-1)
        logloss = eval(trainfea, trainlabel, testfea, testlabel, md)

    return logloss


import lightgbm as lgb
def modelGBM(trainfea, trainlabel, testfea, testlabel, cv=False):
    print("Model GBM...")
    if not cv:
        md = lgb.LGBMClassifier(boosting_type='gbdt',
                                num_leaves=200, #100,
                                max_depth=-1,
                                learning_rate=0.03, #0.05,#0.1,
                                n_estimators=500,
                                max_bin=255, #max_bin=100,#
                                subsample_for_bin=50000,
                                objective='binary', min_split_gain=0,
                                min_child_weight=5, min_child_samples=20,
                                subsample=1, subsample_freq=1,
                                colsample_bytree=1, reg_alpha=0, reg_lambda=0,
                                scale_pos_weight=1, is_unbalance=False,
                                seed=0, nthread=-1, silent=True, sigmoid=1.0,
                                drop_rate=0.1, skip_drop=0.5, max_drop=50,
                                uniform_drop=False, xgboost_dart_mode=False,
                                random_state=1
                                )

        testloss = eval(trainfea, trainlabel, testfea, testlabel, md)

    else:
        #grid search
        #print("GBM CV")
        estimator = lgb.LGBMClassifier(n_jobs = -1, random_state=1)

        param_grid = {
            # default
            # boosting_type='gbdt', num_leaves=31, max_depth=-1,
            # learning_rate=0.1, n_estimators=100, subsample_for_bin=200000,
            # objective=None, class_weight=None, min_split_gain=0.0,
            # min_child_weight=0.001, min_child_samples=20, subsample=1.0,
            # subsample_freq=0, colsample_bytree=1.0,
            # reg_alpha=0.0, reg_lambda=0.0, random_state=None,
            # n_jobs=-1, silent=True, importance_type='split', **kwargs
            #"max_depth": [5, 15],
            "num_leaves":[100, 200, 300],# ,300],
            'n_estimators': [500],
            'learning_rate': [0.05],
            #"reg_alpha":[0.001, 0.01, 0.1, 1]
            "n_jobs": [-1],
            "random_state": [1]
        }

        gbm = GridSearchCV(estimator, param_grid, verbose=50, n_jobs=-1, cv=5)
        gbm.fit(trainfea, trainlabel)
        print('GBM best parameters:', gbm.best_estimator_)

        prod = gbm.best_estimator_
        #print(prod)
        # fit with all training data, no CV
        prod.fit(trainfea, trainlabel)
        trainloss = prod.score(trainfea, trainlabel)
        trainpred = prod.predict_proba(trainfea)
        print("training loss: ", trainloss)
        testpred = prod.predict_proba(testfea)
        testloss = metrics.log_loss(testlabel, testpred)

    print("test loss: ", testloss)
    return testloss


############# Main proc############

DEVELOPMENT = False #True
if DEVELOPMENT:
    devprep()

cvfold =['./1/','./2/','./3/'] #,'./' ]

# config
prepprocs =[ preproc0 ]
models = [ modelGBM
           #modelRF
          ]

perfs = []

# global config
label = ['loan_status']

BYPASSPROC = True

for i in range(len(cvfold)):
    print("Processing fold: ", i)
    # read data
    if BYPASSPROC:
        if os.path.isfile(cvfold[i] + "/trainproc.csv"):
            traindata = pd.read_csv(cvfold[i] + "/trainproc.csv")
        else:
            continue
        if os.path.isfile(cvfold[i] + "/testproc.csv"):
            testdata = pd.read_csv(cvfold[i] + "/testproc.csv")
        else:
            continue
    else:
        if os.path.isfile(cvfold[i] + "/train.csv"):
            traindata = pd.read_csv(cvfold[i] + "/train.csv")
        else:
            continue
        if os.path.isfile(cvfold[i] + "/test.csv"):
            testdata = pd.read_csv(cvfold[i] + "/test.csv")
        else:
            continue

    for prep in prepprocs:
        #traindata, testdata = prep(traindata.loc[0:1000], testdata.loc[0:1000])
        if not BYPASSPROC:
            traindata, testdata = prep(traindata, testdata)
            traindata.to_csv(cvfold[i] + "/trainproc.csv", index=False)
            testdata.to_csv(cvfold[i] + "/testproc.csv", index=False)

        xfeat = [x for x in list(traindata) if x not in ['id', 'loan_status']]
        trainfea = traindata[xfeat]
        print("train feat: ", trainfea.shape)
        trainlabel = traindata[label].values.ravel()
        print("train label: ", trainlabel.shape)

        testfea = testdata[xfeat]
        print("test feat: ", testfea.shape)
        testlabel = testdata[label].values.ravel()
        print("test label: ", testlabel.shape)

        for model in models:
            perf = model(trainfea, trainlabel, testfea, testlabel, cv=False) #True)
            #print(model)
            print("model: ",model, " , perf = ", perf)
            perfs.append({model:perf})

print("final: ", perfs)

print("Time used: %s seconds!" % (time.time() - start_time))
print("Done!")