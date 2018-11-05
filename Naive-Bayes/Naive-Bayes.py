# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 09:25:37 2018

@author: GEAR
"""
import numpy as np

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],				#切分的词条
				['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
				['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
				['stop', 'posting', 'stupid', 'worthless', 'garbage'],
				['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
				['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1] # 1代表侮辱， 0代表非侮辱
    return postingList, classVec

def creatVocabList(dataSet):
    '''
    将切分的词条整理成不重复的词汇表
    '''
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)  #取并集
    return list(vocabSet)

def setOfWordsVec(vocabList, inputSet):
    '''
    将输入的inputSet数据转换为词汇向量
    '''
    returnVec = [0] * len(vocabList)  # 返回的词汇向量
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('the word %s is not in my vocabulary!' % word)
    return returnVec

def trainNaiveBayes(trainData, trainLabels):
    '''
    trainData   为一个二维数组，每一行表示一个数据（returnVec)
    trainlabels 训练类别标签
    '''
    numTrainDocs = len(trainData) # 计算训练文档个数
    numWords = len(trainData[0]) # 计算词条向量的维度
    pAbusive = sum(trainLabels) / float(numTrainDocs) # 文档属于侮辱类的概率
    
    p0Num = np.ones(numWords)   #  
    p1Num = np.ones(numWords)   #
    p0Denom = 2.0; p1Denom = 2.0 # 分母初始化为1
    
    for i in range(numTrainDocs):
        if trainLabels[i] == 1:     #统计属于侮辱类的条件概率数据，P(w0|1),P(w1|1)....，相当于求p(AB)
            p1Num += trainData[i]
            p1Denom += sum(trainData[i])
            
        else:
            p0Num += trainData[i]
            p0Denom += sum(trainData[i])
    p1Vect = np.log(p1Num / p1Denom)   # 每个单词属于1的概率 P(1|w)...相当于p(AB)/p(B) = p(A|B)
    p0Vect = np.log(p0Num / p0Denom)   # 每个单词属于2的概率 P(0|w)...
    
    
    return p0Vect, p1Vect, pAbusive

def NaiveBaysClassify(VectTest, p0Vect, p1Vect, pAbusive):
    '''
    VectTest为测试句子向量
    '''
    # 这里 VectTest * p1Vect计算得到一个向量，向量元素再累乘
    p1 = sum(VectTest * p1Vect)  + np.log(pAbusive)
    p0 = sum(VectTest * p0Vect)  + np.log(1 - pAbusive)
    
    print('p0:',p0)
    print('p1:',p1)
    if p1 > p0:
        return 1  # 属于侮辱类
    else:
        return 0  # 属于非侮辱类
    
def testNativebaysClassify():
    # 生成所要统计文档的词汇列表和对应的类别
    postingList, classVect = loadDataSet()  
    # 统计文档中所有的词汇并生成词汇向量
    myVocabList = creatVocabList(listPosts)
    trainData = []
    # 统计文档中词汇在词汇表中的位置
    for sentence in postingList:
        trainData.append(setOfWordsVec(myVocabList, sentence))
    p0Vect, p1Vect, pAb = trainNaiveBayes(np.array((trainData)), np.array((classVect)))
    
    testDoc1 = ['stupid']
    # 将测试文档转换为词汇向量形式，统计其中每个单词在词汇表中出现的次数和对应的位置
    testWordVect = setOfWordsVec(myVocabList, testDoc1)
    if  NativeBaysClassify(testWordVect, p0Vec, p1Vec, pAb):
        print(testDoc1,'属于侮辱类')										#执行分类并打印分类结果
    else:
        print(testDoc1,'属于非侮辱类')										#执行分类并打印分类结果
        

if __name__ == '__main__':
    testNativebaysClassify()
    
    
    

        