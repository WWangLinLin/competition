# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 21:02:48 2019

@author: Administrator
"""


from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable 
import torch.optim as optim
import torchvision.models as models
import torchvision
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
import os
import copy, time
import string

"""
定义网络
"""
class ConvNet(nn.Module):
      
      def __init__(self):
            super(ConvNet, self).__init__()
            self.conv =nn.Sequential(
                        nn.Conv2d(3, 32, kernel_size=4, stride=1, padding=2), # in:(bs,3,30,150)
                        nn.BatchNorm2d(32),
                        nn.LeakyReLU(0.2, inplace=True),     
                        nn.MaxPool2d(kernel_size=2),        # out:(bs,32,15,75)
                        
                        nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=2),
                        nn.BatchNorm2d(64),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.MaxPool2d(kernel_size=2),        # out:(bs,64,8,38)
                        
                        nn.Conv2d(64, 64, kernel_size=3 ,stride=1, padding=1),
                        nn.BatchNorm2d(64),
                        nn.LeakyReLU(0.2, inplace=True),     
                        nn.MaxPool2d(kernel_size=2)         # out:(bs,64,4,19)
                    )
      
            self.fc1 = nn.Linear(64*4*19, 500)
            self.fc2 = nn.Linear(500,5*62)
      
      def forward(self, x):
            x = self.conv(x)
            print(x.size())
            x = x.view(x.size(0), -1)    # reshape to (batch_size, 64 *4 * 19)
            
            output = self.fc1(x)
            output = self.fc2(output)
            output=output.view(x.size(0)*5,-1)
            
            return output#(bs,5,62)
net = ConvNet()

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)#优化器
loss_func = nn.CrossEntropyLoss()#损失函数

"""
读取图片
"""
import random
fileDir="E:/train"
pathDir = os.listdir(fileDir)    #取图片的原始路径
filenumber=len(pathDir)
rate=1   #自定义抽取图片的比例，比方说100张抽10张，那就是0.1   100张抽30张
picknumber=int(filenumber*rate) #按照rate比例从文件夹中取一定数量图片
sample = random.sample(pathDir, picknumber)
#print(sample)
#print(len(sample))
filenames=np.array([])
filenames=np.append(filenames,sample)
print(sample[0])
print(filenames)
labels=np.array([])
for i in range(filenames.shape[0]):
    la=filenames[i][0:5]
    #print(la)
    labels=np.append(labels,la)
print('labels',labels)

"""
#把0-9，a-z，A-Z用数字表示出来
"""
label_dict={}
characters = string.digits + string.ascii_lowercase+string.ascii_uppercase
for i,x in enumerate(characters):
    label_dict[x]=i
#print(label_dict)
label_dict=label_dict
print(label_dict)#label_dict是{'0': 0, '1': 1, '2': 2,。。。。。}

"""
图片名字的准备即label  label也应该是19800
"""
number_labels=np.array([])
labels=labels[0:19800]
for i in range(19800):
    for j in labels[i]:
        number_labels=np.append(number_labels,label_dict[j])
print(number_labels)#得到的是把字母转换成数字之后所有的
#所以说number_labels是一个一维的，99000个一维的

"""
图片的准备以及训练   #   用了19800张
"""
epoches=30
for epoch in range(epoches):
    print("-----------",epoch)
    pred=np.array([])
    for j in range(198):
        f=0
        for i in range(100):
            f=f+1
            img_path='E:/train'+'/'+filenames[j*100+i]#   用了19900张
            image = Image.open(img_path)
            transform=transforms.ToTensor()
            if f==1:
                images=transform(image)
                images=images.view(1,-1)
            else:
                image=transform(image)
                image=image.view(1,-1)
                images=torch.cat((images,image),0)

        print(images.shape)#torchsize[100,13500] 有200张图片
        images=images.view(100,3,30,150)#500张30*30的图片
        labelss=number_labels[j*500:(j+1)*500]
        labelss=torch.Tensor(labelss).type(torch.LongTensor)
        #inputs=Variable()
        #labelss=Variable(labelss)
        #print('inputs',inputs)
        output=net(images)
        loss=loss_func(output,labelss)
        optimizer.zero_grad()
        loss.backward()#反向传播

        optimizer.step()#更新权重
        prediction = torch.max(F.softmax(output), 1)[1]

        pred_y = prediction.data.numpy().squeeze()
            #print(pred_y)#预测出来的y,是数字的形式
        pred=np.append(pred,pred_y)
        print(pred.shape)
    print(pred)
    p=0
    for k in range(len(pred)):
        if pred[k]==number_labels[k]:
                p=p+1
    print('正确率：',p/len(pred))

"""
将数字转化成字母
"""
def val(va):
    
    for key, val in label_dict.items():
        if val == va:
            return key

testfileDir='E:/test'
list1=[]
for i in range(20000):
    list1.append(("%d%s"%(i,'.jpg')))
print(list1)

testresults=np.array([])
c=0
for i in range(10):
        print(i)
        for k in range(20):
            t=0
            for l in range((i*2000+100*k),(i*2000+100*(k+1))):
                testimg_path=testfileDir+'/'+list1[l]
                testimage=Image.open(testimg_path)
                transform=transforms.ToTensor()
                t=t+1
                c=c+1
                print("-------------",c)
                print(t)
                if t==1:
                    testimages=transform(testimage)
                    testimages=testimages.view(1,-1)
                else:
                    testimage=transform(testimage)
                    testimage=testimage.view(1,-1)
                    testimages=torch.cat((testimages,testimage),0)
            print(testimages.shape)
            testimages=testimages.view(100,3,30,150)
            print(testimages.shape)
            #testinputs=Variable(testimages)
            print(testimages.shape)
            testoutput=net(testimages)
            #print(testoutput)   
            testprediction = torch.max(F.softmax(testoutput), 1)[1]
            testpred_y = testprediction.data.numpy().squeeze()
            #print(testpred_y)
            num=np.array([])
            for k in range(len(testpred_y)):
                num=np.append(num,val(testpred_y[k]))#num是将得到的数字转换为各个标签，得到一个一维numpy数组    
            #print(num)#这里是得到的是一维的所有的标签
            num=num.reshape(-1,5)
            #print(num)
            results=np.array([])
            for j in range(len(num)):
                for q in range(4):
                    if q==0:
                        result=num[j][q]+num[j][q+1]
                    else:
                        result=result+num[j][q+1]
                #print(result)#最后得到的每张图片的标签
                results=np.append(results,result)#一共是100张                
            print(results.shape)#得到的是100张图片
            print(results)
            testresults=np.append(testresults,results)
print(testresults.shape) 

idd=[]  
for j in range(0,20000):
    idd.append(j)

#保存文件
df=pd.DataFrame({'id':idd,'y':testresults})
#dataframe.to_csv("test.csv",index=False,sep=',')
df.to_csv('E:/a4.csv',index=False,sep=',')




