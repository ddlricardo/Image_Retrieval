import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
import threading as td
import pickle

import nus_dataset
import config

traindata = nus_dataset.traindata
#traindata = preprocessing.scale(traindata)
trainlabel = nus_dataset.trainlabel
testdata = nus_dataset.testdata
#testdata = preprocessing.scale(testdata)
testlabel = nus_dataset.testlabel
dbdata = nus_dataset.dbdata
#dbdata = preprocessing.scale(dbdata)
dblabel = nus_dataset.dblabel
groundtruth = nus_dataset.groundtruth

sem = td.Semaphore(config.num_thread)
svms = []

result = []
for i in range(10):
	result.append([])
prediction = []
temp = [0] * 100000
for i in range(2000):
	prediction.append(temp)

def runner(i):
	sem.acquire()
	print("learn begin %s" % i)
	clf = svm.LinearSVC()
	clf = clf.fit(traindata, trainlabel[i])
	svms.append((i, clf))
	result[i] = clf.predict(testdata)
	print("label %s done\n%s"
	 % (i, metrics.classification_report(testlabel[i], result[i])))
	#print metrics.confusion_matrix(testlabel[i], result)
	sem.release()


ts = []

for i in range(10):
	t = td.Thread(target=runner, args=(i,))
	t.start()
	ts.append(t)

for t in ts: t.join()
s = pickle.dumps(svms)

open("svm_dump.bin", 'w').write(s)

for i in range(10):
	for j in range(2000):
		for k in range(100000):
			if result[i][j] == dblabel[i][k] and result[i][j] == 1:
				prediction[j][k] = 1

count = 0
right = 0
for i in range(2000):
	for j in range(100000):
		count += 1
		if prediction[i][j] == groundtruth[i][j]:
			right += 1
print float(right * 1.0 / count)
