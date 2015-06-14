import numpy as np
from sklearn import svm
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing
import threading as td
import cPickle as pickle
import copy
import os

import wiki_dataset
import config

ap = []
prediction = []

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
    config.goto_wiki_dataset()

    traindata = wiki_dataset.traindata
    trainlabel = wiki_dataset.trainlabel

    testdata = wiki_dataset.testdata
    testlabel = wiki_dataset.testlabel

    groundtruth = wiki_dataset.groundtruth

    svms = []
    ok = 0
    if os.path.exists(dump_name):
        print "load model from dump file %s" % dump_name
        ok = 1
        svms = pickle.load(open(dump_name))

    global result, dbresult
    result = []
    dbresult = []
    for i in range(10):
        result.append([])
        dbresult.append([])

    def runner(i):
        print("learn begin %s" % i)

        if ok:
            clf = svms[i][1]
        else:
            clf = trainer(traindata, trainlabel == (i+1))
            svms.append((i, clf))

        result[i] = predictor(clf, testdata, testlabel == (i+1))
        #dbresult[i] = predictor(clf, traindata, trainlabel == (i+1))
        dbresult[i] = (trainlabel == (i+1))

        print("learn done %s" % i)

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
    global prediction
    prediction = relat_calc(result, dbresult)


    print "calc mAP"
    global ap
    ap = []

    def runner2(i):
        n = prediction.shape[0]
        m = n / config.num_thread +1
        l = i*m
        r = min(l+m, n)
        global ap

        for i in range(l,r):
            #print i
            answer = sorted(enumerate(prediction[i]), key=lambda d:d[1], reverse=True)
            apsum = float(0)
            rightsum = 0
            for j in range(prediction.shape[1]):
                if groundtruth[i][answer[j][0]] == 1:
                    rightsum += 1
                    apsum += rightsum * 1.0 / (j + 1)
            ap.append(apsum / sum(groundtruth[i]))


    myrunner(runner2)
    print 'MAP = ' + str(sum(ap) / len(ap))
    print len(ap)

    count = 0
    right = 0

    groundtruth = np.array(groundtruth)
    count = groundtruth.size
    right = np.sum(groundtruth == prediction)

    print float(right * 1.0 / count)
