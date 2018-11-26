# -*- coding:utf-8 -*-
import numpy as np
class NBayes(object):
    def __init__(self,tf_idf = False):
        self.vocabulary = [] # �ʵ�
        self.idf = 0 # Ȩֵ����
        self.tf = 0 
        self.tdm = 0 # P(x|yi)
        self.Pcates = {} # P{yi}
        self.labels = []
        self.doclength = 0
        self.vocablen = 0
        self.testset = 0 # ���Լ�
        self.tf_idf = tf_idf # �Ƿ���� idf Ȩֵ����
        
    def train_set(self,trainset,classVec): # ע�� classVec ֻ�ܴ� 0-n ���ֱ�ʾ
        self.cate_prob(classVec)
        self.doclength = len(trainset)
        tempset = set()
        [tempset.add(word) for doc in trainset for word in doc] # �ʻ㼯��
        self.vocabulary = list(tempset)
        self.vocablen = len(self.vocabulary)
        if self.tf_idf:
            self.calc_tdidf(trainset)
        else:
            self.calc_wordfreq(trainset)
        self.build_tdm() # ����ÿ�����Ӧ�ʵĸ��� P(x|yi)
        
    def cate_prob(self,classVec): # ���� P(yi)
        self.labels = classVec
        labeltemps = set(self.labels)
        for labeltemp in labeltemps:
            self.Pcates[labeltemp] = float(self.labels.count(labeltemp)) / float(len(self.labels))

    def calc_wordfreq(self,trainset): # �����Ƶ
        self.idf = np.zeros([1,self.vocablen]) # ÿ���ʳ��ֵ����´���
        self.tf = np.zeros([self.doclength,self.vocablen])
        for indx in xrange(self.doclength):
            for word in trainset[indx]:
                self.tf[indx,self.vocabulary.index(word)] += 1
            
            for signleword in set(trainset[indx]):
                self.idf[0,self.vocabulary.index(signleword)] += 1
    
    def build_tdm(self):
        self.tdm = np.zeros([len(self.Pcates),self.vocablen])
        sumlist = np.zeros([len(self.Pcates),1]) # ͳ��ÿ��������ܴ�
        for indx in xrange(self.doclength):
            self.tdm[self.labels[indx]] += self.tf[indx] # ÿһ���Ĵ�Ƶ�ӽ�ȥ��Ӧ���
            sumlist[self.labels[indx]] = np.sum(self.tdm[self.labels[indx]])
        
        self.tdm = self.tdm / sumlist # ���� P(x|yi)
        
    def map2vocab(self,testdata):
        self.testset = np.zeros([1,self.vocablen])
        for word in testdata:
            self.testset[0,self.vocabulary.index(word)] += 1
            
    def predict(self,testdata):
        self.map2vocab(testdata)
        if np.shape(self.testset)[1] != self.vocablen:
            print "testset error"
            exit(0)
        
        predvalue = 0
        predclass = ""
        for tdm_vect,keyclass in zip(self.tdm,self.Pcates):
            temp = np.sum(self.testset * tdm_vect * self.Pcates[keyclass])
            if temp > predvalue:
                predvalue = temp
                predclass = keyclass
        
        return predclass
        
    def calc_tdidf(self,trainset):
        self.idf = np.zeros([1,self.vocablen])
        self.tf = np.zeros([self.doclength,self.vocablen])
        for indx in xrange(self.doclength):
            for word in trainset[indx]:
                self.tf[indx,self.vocabulary.index(word)] += 1
            self.tf[indx] = self.tf[indx]/float(len(trainset[indx]))
            for signleword in set(trainset[indx]):
                self.idf[0,self.vocabulary.index(signleword)] += 1
                
        self.idf = np.log(float(self.doclength)/self.idf)
        self.tf = np.multiply(self.tf,self.idf)

if __name__ == "__main__":
    trainset = [["��","�ǳ�","��ˬ"],["��","�ǳ�","����"]] # ����ѵ����
    classVec = [0,1] # ���
    nb = NBayes(tf_idf = True)
    nb.train_set(trainset,classVec)
    print nb.predict(trainset[1])
