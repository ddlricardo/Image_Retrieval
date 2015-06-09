import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing

import config
config.goto_nus_dataset()

trainid = []
traindata = []
trainlabel = []
dbid = []
dbdata = []
dblabel = []
testid = []
testdata = []
testlabel = []

for i in range(10):
	trainlabel.append([])
	dblabel.append([])
	testlabel.append([])

fin = open('trainid.txt', 'r')
raw_data = fin.readlines()
for temp in raw_data:
	trainid.append(temp.strip())
fin.close()

fin = open('testid.txt', 'r')
raw_data = fin.readlines()
for temp in raw_data:
	testid.append(temp.strip())
fin.close()

fin = open('databaseid.txt', 'r')
raw_data = fin.readlines()
for temp in raw_data:
	dbid.append(temp.strip())
fin.close()

fin1 = open('traindata_image.txt', 'r')
fin2 = open('traindata_text.txt', 'r')
raw_data1 = fin1.readlines()
raw_data2 = fin2.readlines()

for i in range(len(raw_data1)):
	temp1 = raw_data1[i].strip().split(' ')
	temp2 = raw_data2[i].strip().split(' ')
	temp = temp1 + temp2
	temp = [int(item) for item in temp]
	traindata.append(temp)
fin1.close()
fin2.close()

fin1 = open('testdata_image.txt', 'r')
fin2 = open('testdata_text.txt', 'r')
raw_data1 = fin1.readlines()
raw_data2 = fin2.readlines()
for i in range(len(raw_data1)):
	temp1 = raw_data1[i].strip().split(' ')
	temp2 = raw_data2[i].strip().split(' ')
	temp = temp1 + temp2
	temp = [int(item) for item in temp]
	testdata.append(temp)
fin1.close()
fin2.close()

fin1 = open('databasedata_image.txt', 'r')
fin2 = open('databasedata_text.txt', 'r')
raw_data1 = fin1.readlines()
raw_data2 = fin2.readlines()
for i in range(len(raw_data1)):
	temp1 = raw_data1[i].strip().split(' ')
	temp2 = raw_data2[i].strip().split(' ')
	temp = temp1 + temp2
	temp = [int(item) for item in temp]
	dbdata.append(temp)
fin1.close()
fin2.close()

iddic = {}

fin = open(r'../ImageId.txt', 'r')
raw_data = fin.readlines()
for i in range(len(raw_data)):
	imgid = raw_data[i].strip()
	iddic[imgid] = i
fin.close()

filenames = [
r'../Concept/Labels_animal.txt',
r'../Concept/Labels_buildings.txt',
r'../Concept/Labels_clouds.txt',
r'../Concept/Labels_grass.txt',
r'../Concept/Labels_lake.txt',
r'../Concept/Labels_person.txt',
r'../Concept/Labels_plants.txt',
r'../Concept/Labels_sky.txt',
r'../Concept/Labels_water.txt',
r'../Concept/Labels_window.txt']

for order in range(10):
	fin = open(filenames[order], 'r')
	raw_data = fin.readlines()
	for i in range(len(trainid)):
		idx = iddic[trainid[i]]
		trainlabel[order].append(raw_data[idx].strip())
	for i in range(len(testid)):
		idx = iddic[testid[i]]
		testlabel[order].append(raw_data[idx].strip())
	for i in range(len(dbid)):
		idx = iddic[dbid[i]]
		dblabel[order].append(raw_data[idx].strip())
	fin.close()

traindata = np.array(traindata)
#traindata = preprocessing.scale(traindata)
trainlabel = np.array(trainlabel)
testdata = np.array(testdata)
#testdata = preprocessing.scale(testdata)
testlabel = np.array(testlabel)
dbdata = np.array(dbdata)
#dbdata = preprocessing.scale(dbdata)
dblabel = np.array(dblabel)
