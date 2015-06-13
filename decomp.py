import numpy as np
from sklearn import decomposition
import pickle


import config

config.goto_wiki_dataset()

files = [
"imfs.bin",
"imfs_vfc7.bin"
]

n = 500


data1 = pickle.load(open("./work/ddl12_temp/pickle/imfs_fc7.bin"))
data2 = pickle.load(open("./work/ddl12_temp/pickle/imfs_vfc7.bin"))


traindata = np.array(data1['feature'])
testdata = np.array(data2['feature'])

pca = decomposition.PCA(n_components=n)
pca.fit(traindata)
traindata = pca.transform(traindata)
testdata = pca.transform(testdata)

data1['feature'] = traindata
data2['feature'] = testdata

open("./work/ddl12_temp/pickle/imfs_fc7_d%d.bin"%n, 'w').write(pickle.dumps(data1))

open("./work/ddl12_temp/pickle/imfs_vfc7_d%d.bin"%n, 'w').write(pickle.dumps(data2))
