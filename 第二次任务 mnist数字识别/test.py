# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 18:59:23 2019

@author: 15218
"""

import torch
import cv2
import torch.nn.functional as F
from mnist手写数字识别 import Cnn  ##重要，虽然显示灰色(即在次代码中没用到)，但若没有引入这个模型代码，加载模型时会找不到模型
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np 
if __name__ =='__main__':    
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    model = Cnn()  
    model.load_state_dict(torch.load('./params.pth'))
    #model = model.to(device)  

    model.eval()    #把模型转为test模式     

    img = cv2.imread("MNIST_data/8/1.jpg")  #读取要预测的图片    

# =============================================================================
#     img = cv2.imread("./2.png")
# =============================================================================
    trans = transforms.Compose([  transforms.ToTensor()])     #经过ToTensor操作图片变成了torch image类型
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#图片转为灰度图，因为mnist数据集都是灰度图    
    img = trans(img)    
    #img = img.to(device)    
    img = img.unsqueeze(0)  #图片扩展多一维,因为输入到保存的模型中是4维的[batch_size,通道,长，宽]，而普通图片只有三维，[通道,长，宽]    #扩展后，为[1，1，28，28]    
    output = model(img)
    print(output)    
    prob = F.softmax(output, dim=1)    
    prob = Variable(prob)    
    prob = prob.cpu().numpy()  #用GPU的数据训练的模型保存的参数都是gpu形式的，要显示则先要转回cpu，再转回numpy模式    
    print("每个数字的预测概率：\n",prob)  #prob是10个分类的概率    
    pred = np.argmax(prob) #选出概率最大的一个    
    print("预测数字为：",pred.item())

