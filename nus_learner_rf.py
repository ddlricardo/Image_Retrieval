import numpy as np
from sklearn import svm
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing
from sklearn import ensemble
import threading as td
import pickle
import copy

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
	prediction.append(copy.copy(temp))

def runner(i):
	sem.acquire()
	print("learn begin %s" % i)
	clf = ensemble.RandomForestClassifier(n_estimators=100)
	clf = clf.fit(traindata, trainlabel[i])
	svms.append((i, clf))
	result[i] = clf.predict_proba(testdata)
	#print("label %s done\n%s"
	# % (i, metrics.classification_report(testlabel[i], result[i])))
	#print metrics.confusion_matrix(testlabel[i], result)
	sem.release()

prate0 = [0.96, 0.92, 0.8, 0.89, 0.93, 0.93, 0.89, 0.81, 0.94, 0.92]
prate1 = [0.87, 0.56, 0.77, 0.66, 0.64, 0.79, 0.87, 0.79, 0.68, 0.74]

ts = []

for i in range(10):
	t = td.Thread(target=runner, args=(i,))
	t.start()
	ts.append(t)

for t in ts: t.join()
s = pickle.dumps(svms)

open("rf_dump.bin", 'w').write(s)

for i in range(10):
	for j in range(2000):
		for k in range(100000):
			if dblabel[i][k] == 1:
				prediction[j][k] += result[i][j][1]

ap = []
for i in range(2000):
	orderdic = {}
	for j in range(100000):
		orderdic[j] = prediction[i][j]
	answer = sorted(orderdic.iteritems(), key=lambda d:d[1], reverse=True)
	apsum = float(0)
	rightsum = 0
	for j in range(100000):
		if groundtruth[i][answer[j][0]] == 1:
			rightsum += 1
			apsum += rightsum * 1.0 / (j + 1)
	ap.append(apsum / sum(groundtruth[i]))
print 'MAP = ' + str(sum(ap) / len(ap))

count = 0
right = 0
for i in range(2000):
	for j in range(100000):
		count += 1
		if prediction[i][j] == groundtruth[i][j]:
			right += 1
print float(right * 1.0 / count)
