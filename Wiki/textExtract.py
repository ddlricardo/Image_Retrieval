# -*- coding: utf8 -*-
#yangyuan
#2015/6/8
#处理数据 


import re
import os
import math
import json

wordlist = []
wordnum = {}
wordno = {}
wno = 0
wf = {}
trdf = {}
tedf = {}
trdocs = 0.0
tedocs = 0.0
datapath = ""


def trainExtract(text):
    text = text.lower()
    txt = re.findall('<text>([\w\W]+)</text>', text)[0]
    words = re.findall('[a-z]{4,}', txt)
    d = {}
    global wordlist, wordnum, trdf, wno, wordno
    t = 1.0 / len(words)
    for w in words:
        if d.has_key(w):
            d[w] += 1
        else:
            d[w] = 1
        if wordnum.has_key(w):
            wordnum[w] += 1
            if d[w] == 1:
                trdf[w] += 1
        else:
            wordnum[w] = 1
            wno += 1
            wordno[w] = wno
            trdf[w] = 1
            wordlist.append(w)
    return d

def testExtract(text):
    text = text.lower()
    txt = re.findall('<text>([\w\W]+)</text>', text)[0]
    words = re.findall('[a-z]{4,}', txt)
    d = {}
    global wordlist, wordnum, tedf
    for w in words:
        if not wordnum.has_key(w):
            continue
        if d.has_key(w):
            d[w] += 1
        else:
            d[w] = 1
            if tedf.has_key(w):
                tedf[w] += 1
            else:
                tedf[w] = 1
    return d

def dealdata():
    global wordlist, wordnum, trdf, wno, wordno, trdocs, tedocs, tedf
    
    folder = datapath + "texts/"
    f = open(datapath+'trainset_txt_img_cat.list', 'r')
    trdata = []
    for line in f:
        trdocs += 1
        t = line.split('\t')
        fname = t[0] + '.xml'
        typeno = int(t[2])
        fp = open(folder + fname, 'r')
        d = trainExtract(fp.read())
        trdata.append((d, typeno))
        fp.close()
    f.close()
    print wno
    
    f = open("train", "w")
    for i in trdata:
        s = str(i[1]) + ' '
        for w in wordlist:
            if not i[0].has_key(w):
                continue
            idf = math.log(trdocs / trdf[w])
            wij = (1 + math.log(i[0][w])) * idf
            s += str(wordno[w])+':'+str(wij) + ' '
        s += '\n'
        f.write(s)   
    f.close()
    f = open(datapath+'/testset_txt_img_cat.list', 'r')
    tedata = []
    for line in f:
        tedocs += 1
        t = line.split('\t')
        fname = t[0] + '.xml'
        typeno = int(t[2])
        fp = open(folder + fname, 'r')
        d = testExtract(fp.read())
        tedata.append((d, typeno))
        fp.close()
    f.close()
    f = open("test", "w")
    for i in tedata:
        s = str(i[1]) + ' '
        for w in wordlist:
            if not i[0].has_key(w):
                continue
            idf = math.log(trdocs / trdf[w])
            wij = (1 + math.log(i[0][w])) * idf
            s += str(wordno[w]) + ':' + str(wij) + ' '
        s += '\n'
        f.write(s)
    f.close()

if __name__ == '__main__':
    fp = open("config.json", "r")
    datapath = json.loads(fp.read())['datapath']
    fp.close()
    print datapath
    dealdata()
        
