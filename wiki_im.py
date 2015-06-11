import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
import threading as td
import pickle

import config

config.goto_wiki_dataset()

data = pickle.load(open("./work/ddl12_temp/pickle/imfs_fc7_d500.bin"))

traindata = np.array(data['feature'])
trainlabel = np.array(data['label'])

testdata = traindata
testlabel = trainlabel

clf = svm.SVC()
clf = clf.fit(traindata, trainlabel)

result = clf.predict(testdata)
print(metrics.classification_report(testlabel, result))

s = pickle.dumps(clf)
open('model/fc7_gs_d500.bin','w').write(s)
