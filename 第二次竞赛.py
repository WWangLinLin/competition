# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 18:50:03 2019

@author: Administrator
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

"""
对数据集缺失值进行处理，用每列的中位数替代缺失值
"""
#缺失值用Nan代替
data = pd.read_csv("train.csv",header=None)
data[[0,1,2,3,4,5,6,7,8,9,10,11,12,13]]=data[[0,1,2,3,4,5,6,7,8,9,10,11,12,13]].replace('?',np.NaN)

#用每一列的均值代替了data中的NaN
data0 = pd.read_csv("train.csv",header=None)
data0[[0,1,2,3,4,5,6,7,8,9,10,11,12,13]]=data0[[0,1,2,3,4,5,6,7,8,9,10,11,12,13]].replace('?',np.NaN)
data0.dropna(inplace=True)
data0.to_csv('jingsai00.csv',index=False,header=None,sep=',')
data1 = pd.read_csv("jingsai00.csv",header=None)
colmean=data1.median()
#print(colmean)#所有列的中位数值
for i in range(14):
    data[i].fillna(colmean[i],inplace=True)
#print(data)
#所以现在的data是正常的数据了，下面就可以进行分类了
data.to_csv('jingsai11.csv',index=False,header=None,sep=',')


"""
构建神经网络
"""
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   
        self.out = torch.nn.Linear(n_hidden, n_output)   

    def forward(self, x):
        x = F.relu(self.hidden(x))      
        x = self.out(x)
        return x
net = Net(n_feature=13, n_hidden=10, n_output=10)

"""
读取训练集数据,并用卡方检验来对训练集数据再一次进行处理
"""
def traindata(path):
    data=np.loadtxt(path,delimiter=",")#训练集数据
    data1=data[:,0:13]
    #print(type(data1[0]))
    data22=data[:,13]
    data1=torch.Tensor(data1)
    #data2=torch.from_numpy(data2)
    data2=torch.Tensor(data22).type(torch.LongTensor)
    x, y = Variable(data1),Variable(data2)
    return x,y,data22     
x,y,data22=traindata('jingsai11.csv')

"""
训练模型
"""
def train(epoch,x,y,data22):
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)#优化器                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
    loss_func = torch.nn.CrossEntropyLoss()#损失函数
    for t in range(epoch):
        out = net(x)                 
        loss = loss_func(out, y)    #计算损失
        optimizer.zero_grad()  
        loss.backward()         #反向传播
        optimizer.step()  #更新权重
        prediction = torch.max(F.softmax(out), 1)[1]
        pred_y = prediction.data.numpy().squeeze()
    
    a=0
    for i in range(len(pred_y)):
        if pred_y[i]==list(data22)[i]:
            a=a+1
    print('在训练集中预测的正确率：',a/7194)
train(10000,x,y,data22)#训练模型

"""
对测试集的缺失值进行处理，用每列的中位数替代缺失值
"""
#用每一列的均值代替了data中的NaN
testdata = pd.read_csv("test.csv",header=None)
testdata[[0,1,2,3,4,5,6,7,8,9,10,11,12]]=testdata[[0,1,2,3,4,5,6,7,8,9,10,11,12]].replace('?',np.NaN)
#print(testdata.isnull().sum())
testdata0 = pd.read_csv("test.csv",header=None)
testdata0[[0,1,2,3,4,5,6,7,8,9,10,11,12]]=testdata0[[0,1,2,3,4,5,6,7,8,9,10,11,12]].replace('?',np.NaN)
testdata0.dropna(inplace=True)
testdata0.to_csv('jingsai000.csv',index=False,header=None,sep=',')
testdata1 = pd.read_csv("jingsai000.csv",header=None)
colmean=testdata1.median()
#print(colmean)#所有列的均值
for i in range(13):
    testdata[i].fillna(colmean[i],inplace=True)
#所以现在的data是正常的数据了，下面就可以进行分类了
testdata.to_csv('jingsai111.csv',index=False,header=None,sep=',')

"""
对测试集进行预测
"""
testdata1=np.loadtxt('jingsai111.csv',delimiter=",")#测试集数据
testdata1=torch.Tensor(testdata1)
testout=net(testdata1)#将数据放到神经网络中
testprediction = torch.max(F.softmax(testout), 1)[1]
testpred_y = testprediction.data.numpy().squeeze()
print('测试集预测的结果：',testpred_y)#得到预测数据

"""
最终得到的预测数据并进行保存
"""
#文档中第一列id
idd=[]  
for j in range(1,len(testpred_y)+1):
    idd.append(j)
#保存文件
import pandas as pd
df=pd.DataFrame({'id':idd,'y':testpred_y})
#dataframe.to_csv("test.csv",index=False,sep=',')
df.to_csv('test_results.csv',index=False,sep=',')


