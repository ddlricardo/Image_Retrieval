import numpy as np
from sklearn import svm
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing
from sklearn import ensemble
import threading as td
import pickle
import copy

import wiki_trainer

def trainerb(traindata, trainlabel):
    clf = ensemble.BaggingClassifier(
        linear_model.LogisticRegression())
    #clf.verbose = 1
    #clf.tol = clf.tol / 10
    clf = clf.fit(traindata, trainlabel)
    return clf


def trainer(traindata, trainlabel):
    clf = linear_model.LogisticRegression(tol = 0.00001, C=1000000)
    #clf.verbose = 1
    #clf.tol = clf.tol / 10
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
#wiki_trainer.run(trainer, predictor, relat_calc2, "lr2.bin", 'both')
maps = {}
max_map = 0
max_rt = 0
#wiki_trainer.run(trainer, predictor, relat_calc, "lr9.bin", 'both', 0.02, False)

for i in range(100):
    rt = i*0.01
    mp = wiki_trainer.run(trainer, predictor, relat_calc, "lr9.bin", 'both', rt, False)
    if mp > max_map:
        max_map = mp
        max_rt = rt
    maps[rt] = mp
    print "now ", max_map, max_rt
