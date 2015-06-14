import pickle
import numpy as np
import config


config.goto_wiki_dataset()

print "data loading"

data = pickle.load(open("./work/ddl12_temp/pickle/imfs_fc7.bin"))

traindata = np.array(data['feature'])
trainlabel = np.array(data['label'])

data = pickle.load(open("./work/ddl12_temp/pickle/imfs_vfc7.bin"))

testdata = np.array(data['feature'])
testlabel = np.array(data['label'])

groundtruth = testlabel.reshape((-1, 1)) == trainlabel.reshape((1, -1))

print "data load done"
