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


sem = td.Semaphore(2)
svms = []

def runner(i):
	sem.acquire()
	print("learn begin %s" % i)
	clf = svm.LinearSVC()
	clf = clf.fit(traindata, trainlabel[i])
	svms.append((i, clf))
	result = clf.predict(testdata)
	print("label %s done\n%s"
	 % (i, metrics.classification_report(testlabel[i], result)))
	print metrics.confusion_matrix(testlabel[i], result)
	sem.release()


ts = []

for i in range(10):
	t = td.Thread(target=runner, args=(i,))
	t.start()
	ts.append(t)
