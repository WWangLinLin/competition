# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 15:58:30 2019

@author: Administrator
"""

import numpy as np
data=np.loadtxt('HTRU_2_train.csv',delimiter=",")#训练集数据
p,q=np.shape(data)
data_property=data[:,0:2]#训练集数据的属性
data_class=data[:,2]#训练集数据的类别

#kNN
def kNN(test,train_property,train_class,k):
    # 测试数据   训练集属性    训练集类别  近邻数
    
    #  #step1计算距离
    m,n=np.shape(train_property)#训练集的行数、列数
    difference=np.tile(test,(m,1))-train_property#按元素求差值
    #np.title(A,reps):构造一个矩阵，将A重复reps次
    #将test行重复m次，列重复一次、
    
    sqd=difference**2#将差值平方
    numSqd=np.sum(sqd,axis=1)#按行累加
    distance=numSqd**0.5#开平方得到距离
    
    #  #step2 对距离进行排序得到索引
    distanceIndex=np.argsort(distance)#返回排序后其中元素在原来位置的索引
    #print(distanceIndex)
    classCount={}
    for i in np.arange(k):
        
        #  #step3 选择k个近邻
        #distanceIndex[i] 例如当k=0时，即找出distanceIndex中的第一个值，第一个值是排好序后最小的距离（在原来位置）的索引
        #train_class[distanceIndex[i]]：由原来位置的索引（distanceIndex[i]）就能得到对应的类别
        classNumber=train_class[distanceIndex[i]]#有了索引distanceIndex[i]就能找到相应位置的类别
        
        #step4：计算k个最近邻中出现各个类别的次数
        classCount[classNumber]=classCount.get(classNumber,0)+1
    
    #  #step 5： 返回出现次数最多的类别标签，即预测的类别
    maxCount=0
    for key,value in classCount.items():
        if value>maxCount:
            maxCount=value
            maxIndex=key
    return maxIndex

test_data=np.loadtxt('HTRU_2_test.csv',delimiter=",")
a,b=np.shape(test_data)
test_array=np.array([])
for i in np.arange(a):#对于测试集中的每一个样本数据
    test_array=np.append(test_array,kNN(test_data[i],data_property,data_class,15))

test_arraya=np.array(test_array,dtype=int)#预测的类别

#文档中第一列id
idd=[]  
for j in range(1,len(test_arraya)+1):
    idd.append(j)

#保存文件
import pandas as pd
df=pd.DataFrame({'id':idd,'y':test_arraya})
#dataframe.to_csv("test.csv",index=False,sep=',')
df.to_csv('tttest.csv',index=False,sep=',')
