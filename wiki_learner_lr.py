import numpy as np
from sklearn import svm
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing
import threading as td
import pickle
import copy

import wiki_trainer

def trainer(traindata, trainlabel):
    clf = linear_model.LogisticRegression()
    clf = clf.fit(traindata, trainlabel)
    return clf

def predictor(clf, testdata, label):
    result = clf.predict_proba(testdata)
    return result

def groundtruth_predictor(clf, testdata, label):
    return label

def relat_calc(la, lb):
    if la.ndim == 4:
        # 由于 LogisticRegression 返回两个数
        #       n:1:10
        la = la[:,:,:,1]
        #       1:m:10

    if lb.ndim == 4:
        lb = lb[:,:,:,1]

    res = np.zeros((la.shape[0], lb.shape[1]))
    for i in range(la.shape[0]):
        res[i] = np.sum(la[i] * lb, axis=2)
    return res

def relat_calc2(la, lb):
    if la.ndim == 4:
        la = la[:,:,:,1]

    if lb.ndim == 4:
        lb = lb[:,:,:,1]

    res = np.zeros((la.shape[0], lb.shape[1]))
    for i in range(la.shape[0]):
        res[i] = 1 - np.prod(1 - la[i] * lb, axis=2)
    return res

#nus_trainer.run(trainer, predictor, relat_calc, "lr2.bin")
wiki_trainer.run(trainer, predictor, relat_calc2, "lr.bin")
