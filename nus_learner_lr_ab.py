import numpy as np
from sklearn import svm
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing
import threading as td
import pickle
import copy

import nus_trainer

def trainer(traindata, trainlabel):
	clf = AdaBoostClassifier(linear_model.LogisticRegression())
	clf = clf.fit(traindata, trainlabel)
	return clf

def predictor(clf, testdata, label):
	result = clf.predict_proba(testdata)
	return label

def relat_calc(la, lb):
	if la.ndim == 4:
		# 由于 LogisticRegression 返回两个数
		#       n:1:10
		la = la[:,:,:,1]
		#       1:m:10
		lb = lb[:,:,:,1]

	res = np.zeros((la.shape[0], lb.shape[1]))
	for i in range(la.shape[0]):
		res[i] = np.sum(la[i] * lb, axis=2)
	return res

def relat_calc2(la, lb):
	if la.ndim == 4:
		la = la[:,:,:,1]
		lb = lb[:,:,:,1]

	res = np.zeros((la.shape[0], lb.shape[1]))
	for i in range(la.shape[0]):
		res[i] = 1 - np.prod(1 - la[i] * lb, axis=2)
	return res

nus_trainer.run(trainer, predictor, relat_calc, "lr_ab.bin")
