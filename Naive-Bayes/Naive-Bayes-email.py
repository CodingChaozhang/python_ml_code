# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 15:55:43 2018

@author: GEAR
"""
import re  #正则表达式
import numpy as np
import random

def textParse(email_path):
    '''
    将邮件中的内容转换为字符列表
    '''
    listOfEmail = re.split(r'\W*', email_path)
    return [tok.lower() for tok in listOfEmail if len(tok) > 2]

def creatVocabList(dataSet):
    '''
    创建词汇表,这里dataSet为二维列表
    '''
    vocabSet = set([])
    for i in dataSet:
        vocabSet = vocabSet | set(i)
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    '''
    将输入的inputSet数据转换为词汇向量
    '''
    returnVect = [0] *len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVect[vocabList.index(word)] = 1
        else:
            print('the word %s not in vocabList' %word)
    return returnVect

def trainNaiveBayes(trainData, trainLabels):
    '''
    trainData   为一个二维数组，每一行表示一个数据（returnVec)
    trainlabels 训练类别标签
    '''
    numTrains = len(trainData)   # 计算训练样本个数
    numWords = len(trainData[0]) # 计算每个样本中单词个数
    # 属于侮辱性样本的概率
    pAbusive = sum(trainLabels) / len(trainLabels)
    p0Num = np.ones(numWords); p1Num = np.ones(numWords)
    p0Sum = 2.0; p1Sum = 2.0
    
    for i in range(numTrains):
        if trainLabels[i] == 1:
            p1Num += trainData[i]
            p1Sum += sum(trainData[i])
        else:
            p0Num += trainData[i]   # 每个单词在非侮辱样本中出现的次数
            p1Sum += sum(trainData[i]) # 非侮辱样本中所有单词出现的次数
    p1Vect = np.log(p1Num / p1Sum) # 每个单词是侮辱性样本的概率p(w|1)
    p0Vect = np.log(p0Num / p0Sum)
    
    return p0Vect, p1Vect, pAbusive

def NaiveBaysClassify(VectTest, p0Vect, p1Vect, pAbusive):
    '''
    对每一个输入的文档VectTest进行预测
    '''
    # 计算该文档属于1的概率
    p1 = sum(VectTest*p1Vect) + np.log(pAbusive)
    # 计算该文档属于0的概率
    p0 = sum(VectTest*p0Vect) + np.log(1 - pAbusive)
    
    if p1 > p0:
        return 1
    else:
        return 0
    
    
def spamclassifyTest(root):
    '''
    对垃圾邮件进行预测
    '''
    docList = []; classList = []; fullText = []
    for i in range(1, 26):                          # 遍历25个txt文档
        wordList = textParse(open(root + 'spam/%d.txt' % i, 'r').read()) #读取每个垃圾邮件
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(1)                        #标记垃圾邮件， 1表示垃圾 邮件
        wordList = textParse(open(root + 'ham/%d.txt' % i, 'r').read())
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(0)
        
    vocabList = creatVocabList(docList)            #创建词汇表
    trainSet = list(range(50)); testSet = []       #创建存储训练集和测试集的索引值的列表
    for i in range(10):     #从50个数据集中随机选取40个训练，10个测试
        randIndex = int(random.uniform(0, len(trainSet)))   # 随机选取索引值
        testSet.append(trainSet[randIndex])                 # 添加测试集的索引值
        del(trainSet[randIndex])                            # 在训练集中删除测试集中对应的元素
    trainArray = []; trainClass = []
    for index in trainSet:
        trainArray.append(setOfWords2Vec(vocabList, docList[index]))
        trainClass.append(classList[index])
    p0Vect, p1Vect, pSpam = trainNaiveBayes(np.array(trainArray), np.array(trainClass))
    errorCount = 0
    for index in testSet:
        wordVect = setOfWords2Vec(vocabList, docList[index])
        if NaiveBaysClassify(wordVect, p0Vect, p1Vect, pSpam) != classList[index]:
            errorCount += 1
            print('分类错误的测试集：', docList[index])
    print('错误率：%.2f%%' % (float(errorCount) / len(testSet) * 100))
    
if __name__ == '__main__':
    root = 'D:/machine learing/python-ml-code/Naive-Bayes/email/'
    spamclassifyTest(root)
    
        
    
        
        