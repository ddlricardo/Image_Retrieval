import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
import threading as td
import pickle

import config

config.goto_wiki_dataset()

data = pickle.load(open("./work/ddl12_temp/pickle/imfs_vfc7_d500.bin"))
clf = pickle.load(open("./model/fc7_gs_d500.bin"))


testdata = np.array(data['feature'])
testlabel = np.array(data['label'])

result = clf.predict(testdata)
print(metrics.classification_report(testlabel, result))
