# -*- coding: utf-8 -*-

#yangyuan

from svmutil import *

datapath = "../"  #where is data? train, test and result

trainy, trainx =  svm_read_problem(datapath + 'train')
### y:label x:feture
count = [0 for i in range(11)]
for i in trainy:
    count[int(i)] += 1
#print count

#m = svm_train(trainy, trainx, '-s 1 -t 3 -b 1')

testy, testx = svm_read_problem(datapath + 'test')

#plabel, p_acc, p_val = svm_predict(testy, testx, m, '-b 1')

fp = open(datapath + "result","r")
content = fp.read()
res = content.strip().split('\n')

labels = res[0].split(' ')
lb = [int(i) for i in range(1, 11)]

a = [0 for i in range(11)]  #correct to k
b = [0 for i in range(11)]  #wrong to k
c = [0 for i in range(11)]  #k to wrong


APsum = 0.0
for i in range(1, len(res)):
    #print i,
    tmp = res[i].strip().split(' ')
    predictlabel = int(tmp[0])
    probility = {i-1:float(tmp[i]) for i in range(1, 11)}
    sp = sorted(probility, key=probility.get, reverse=True) #sort by probility

    reallb = int(testy[i-1])
    if predictlabel == reallb:
        APsum += 1
        a[reallb] += 1
    else:
        b[predictlabel] += 1
        c[reallb] += 1
        num = 0
        for i in range(10):
            if lb[sp[i]] != reallb:
                num += count[reallb]
            else:
                break
        ap = 0
        tmpc = count[reallb]
        for i in range(1, tmpc+1):
            ap += float(i)/ (tmpc+i)
        ap /= tmpc
        APsum += ap

            
MAP = APsum / (len(res) - 1)
print "MAP =",MAP
#print a,b,c
sp, sr = 0, 0
for i in range(1,11):
    sp += float(a[i]) / (a[i] + b[i])
    sr += float(a[i]) / (a[i] + c[i])
print "P =", sp / 10
print "R =", sr / 10 

