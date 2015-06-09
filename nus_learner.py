import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
import threading as td

import nus_dataset

traindata = nus_dataset.traindata
#traindata = preprocessing.scale(traindata)
trainlabel = nus_dataset.trainlabel
testdata = nus_dataset.testdata
#testdata = preprocessing.scale(testdata)
testlabel = nus_dataset.testlabel
dbdata = nus_dataset.dbdata
#dbdata = preprocessing.scale(dbdata)
dblabel = nus_dataset.dblabel

def runner(i):
	print("learn begin %s" % i)
	clf = svm.SVC()
	clf = clf.fit(traindata, trainlabel[i])
	result = clf.predict(testdata)
	print(metrics.classification_report(testlabel[i], result))


ts = []

for i in range(10):
	t = td.Thread(target=runner, args=(i,))
	t.start()
	ts.append(t)
