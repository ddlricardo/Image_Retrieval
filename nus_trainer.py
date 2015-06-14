import numpy as np
from sklearn import svm
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing
import threading as td
import pickle
import copy
import os

import nus_dataset
import config

def myrunner(func):
    sem = td.Semaphore(config.num_thread)

    def wrapper(i):
        sem.acquire()
        try:
            func(i)
        except Exception as e:
            raise
        finally:
            sem.release()

    ts = []
    for i in range(10):
    	t = td.Thread(target=wrapper, args=(i,))
    	t.start()
    	ts.append(t)

    for t in ts: t.join()

def run(trainer, predictor, relat_calc, dump_name):
    config.goto_nus_dataset()
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

    svms = []
    ok = 0
    if os.path.exists(dump_name):
        print "load model from dump file %s" % dump_name
        ok = 1
        svms = pickle.load(open(dump_name))

    result = []
    dbresult = []
    for i in range(10):
    	result.append([])
    	dbresult.append([])
    prediction = []
    temp = [0] * 100000
    for i in range(2000):
    	prediction.append(copy.copy(temp))

    def runner(i):
    	print("learn begin %s" % i)

        if ok:
            clf = svms[i][1]
        else:
            clf = trainer(traindata, trainlabel[i])
    	    svms.append((i, clf))

        result[i] = predictor(clf, testdata, testlabel[i])
        dbresult[i] = predictor(clf, dbdata, dblabel[i])

    print "training"
    myrunner(runner)

    svms = sorted(svms, key=lambda x:x[0])
    s = pickle.dumps(svms)
    open(dump_name, 'w').write(s)

    result = np.array(result)
    dbresult = np.array(dbresult)

    result = np.rollaxis(result, 0, 2)
    dbresult = np.rollaxis(dbresult, 0, 2)

    result.shape = (result.shape[0],) + (1,) + (result.shape[1:])
    dbresult.shape = (1,) + (dbresult.shape[0],) + (dbresult.shape[1:])

    print "calc relation %s %s" % (str(result.shape), str(dbresult.shape))
    prediction = relat_calc(result, dbresult)


    print "calc mAP"
    ap = []

    def runner2(i):
        n = prediction.shape[0]
        m = n / config.num_thread +1
        l = i*m
        r = min(l+m, n)

        for i in range(l,r):
            if i%10 == 0:
                print i
        	answer = sorted(enumerate(prediction[i]), key=lambda d:d[1], reverse=True)
        	apsum = float(0)
        	rightsum = 0
        	for j in range(100000):
        		if groundtruth[i][answer[j][0]] == 1:
        			rightsum += 1
        			apsum += rightsum * 1.0 / (j + 1)
        	ap.append(apsum / sum(groundtruth[i]))

    myrunner(runner2)
    print 'MAP = ' + str(sum(ap) / len(ap))

    count = 0
    right = 0

    groundtruth = np.array(groundtruth)
    count = groundtruth.size
    right = np.sum(groundtruth == prediction)

    print float(right * 1.0 / count)
