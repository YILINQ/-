import pickle
from collections import defaultdict
import os
import re
import jieba
import codecs
import sys
import matplotlib.pyplot as plt
import pandas as pd
df_sentiment = pd.DataFrame()

def sent2word(sentence):
    seglist = jieba.cut(sentence)
    segresult = []
    for w in seglist:
        segresult.append(w)
    stopwords = readLines('stop_words.txt')
    newsent = []
    for w in segresult:
        if w+'\n' not in stopwords:
            newsent.append(w)
    return newsent

def eachFile(filename):
    pathDir = os.listdir(filename)
    child = []
    for aD in pathDir:
        child.append(os.path.join('%s/%s' % (filename, aD)))
    return child

def readLines(filename):
    fopen = open(filename, 'r')
    data=[]
    for x in fopen.readlines():
        if x.strip() != '':
            data.append(x.strip())
    fopen.close()
    return data

def readLines2(filename):
    fopen = open(filename, 'r')
    data=[]
    for x in fopen.readlines():
        if x.strip() != '':
            data.append(x.strip())
    fopen.close()
    return data

def words():
    senL = readLines2('BosonNLP_sentiment_score.txt')
    senD = defaultdict()
    for s in senL:
        senD[s.split(' ')[0]] = s.split(' ')[1]
    noL = readLines2('notDict.txt')
    degreeeL = readLines2('degreeDict.txt')
    degreeD = defaultdict()
    for d in degreeeL:
        degreeD[d.split(',')[0]] = d.split(',')[1]
    return senD, noL, degreeD

def classifyWords(wordDict, senD, noL, degreeD):
    senW = defaultdict()
    notW = defaultdict()
    degreeW = defaultdict()
    for word in wordDict.keys():
        if word in senD.keys() and word not in noL and word not in degreeD.keys():
            senW[wordDict[word]] = senD[word]
        elif word in noL and word not in degreeD.keys():
            notW[wordDict[word]] = -1
        elif word in degreeD.keys():
            degreeW[wordDict[word]] = degreeD[word]
    return senW, notW, degreeW

def scoreSent(senW, notW, degreeW, segResult):
    W = 1
    score = 0
    senLoc = list(senW.keys())
    notLoc = notW.keys()
    degreeLoc = degreeW.keys()
    senloc = -1

    for i in range(0, len(segResult)):
        if i in senLoc:
            senloc += 1
            score += W * float(senW[i])
            if senloc < len(senLoc) - 1:
                for j in range(senLoc[senloc], senLoc[senloc + 1]):
                    if j in notLoc:
                        W *= -1
                    elif j in degreeLoc:
                        W *= float(degreeW[j])
        if senloc < len(senLoc) - 1:
            i = senLoc[senloc + 1]
    return score

def listToDist(wordlist):
    data={}
    for x in range(0, len(wordlist)):
        data[wordlist[x]] = x
    return data

def runplt():
    plt.figure()
    plt.title('test')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis([0, 1000, -10, 10])
    plt.grid(True)
    return plt

if __name__ == "__main__":
    a, b, c = words()
    word_l = readLines2('data.txt')
    word_d = listToDist(word_l)
    result = {}
    for s in word_d:
        word = s
        ttt = sent2word(s)
        aa, bb, cc = classifyWords(word_d, a, b, c)
        total = 0
        for x in ttt:
            total += scoreSent(aa, bb, cc, x)
        result[word] = total
    print(result)
