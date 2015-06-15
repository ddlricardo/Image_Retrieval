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

def load_libsvm_format(fname):
    ls = open(fname).readlines()
    maxdim = 0
    fs = []
    label = []
    for s in ls:
        ss = s.strip().split(' ')
        label.append(int(ss[0]))
        d = {}
        for p in ss[1:]:
            a,b = p.split(':')
            d[int(a)] = float(b)
            maxdim = max(maxdim, int(a))
        fs.append(d)
    ifs = []
    for dd in fs:
        f = [0]*(maxdim+1)
        for k,v in dd.items():
            f[k] = v
        ifs.append(f)
    return np.array(ifs), np.array(label)

traindata2, trainlabel2 = load_libsvm_format("./wiki_text/train")
testdata2, testlabel2 = load_libsvm_format("./wiki_text/test")

def check(l1,l2):
    if np.sum(l1 == l2) != l1.size:
        raise Exception("error: label not equal")

check(trainlabel2, trainlabel)
check(testlabel2, testlabel)

print "data load done"
