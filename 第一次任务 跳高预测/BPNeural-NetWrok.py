# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np

#运动员各项指标
X = np.array([
    3.2, 9.6, 3.45, 2.15, 140, 2.8, 11.0, 50, 3.2, 10.3, 3.75, 2.2, 120, 3.4,
    10.9, 70, 3.0, 9.0, 3.5, 2.2, 140, 3.5, 11.4, 50, 3.2, 10.3, 3.65, 2.2,
    150, 2.8, 10.8, 80, 3.2, 10.1, 3.5, 2, 80, 1.5, 11.3, 50, 3.4, 10.0, 3.4,
    2.15, 130, 3.2, 11.5, 60, 3.2, 9.6, 3.55, 2.1, 130, 3.5, 11.8, 65, 3.0,
    9.0, 3.5, 2.1, 100, 1.8, 11.3, 40, 3.2, 9.6, 3.55, 2.1, 130, 3.5, 11.8, 65,
    3.2, 9.2, 3.5, 2.1, 140, 2.5, 11.0, 50, 3.2, 9.5, 3.4, 2.15, 115, 2.8,
    11.9, 50, 3.9, 9.0, 3.1, 2.0, 80, 2.2, 13.0, 50, 3.1, 9.5, 3.6, 2.1, 90,
    2.7, 11.1, 70, 3.2, 9.7, 3.45, 2.15, 130, 4.6, 10.85, 70
]).reshape(14, 8)
#运动员的跳高成绩
Y = np.array([
    2.24, 2.33, 2.24, 2.32, 2.2, 2.27, 2.2, 2.26, 2.2, 2.24, 2.24, 2.2, 2.2,
    2.35
]).reshape(14, 1)
#输入层、隐藏层、输出层的结点个数
layer = [8, 6, 1]
#学习率
alpha = 0.01


#sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


#sigmoid函数的导数
def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.where(x<0,0,x)

def relu_derivative(x):
    return np.where(x<0,0,1)

class BP():
    def __init__(self, X, Y, layer,num=5000):
        self.sample_num, self.input_num = np.shape(X)
        np.random.seed(1)
        #样本X,Y值
        self.train_input = X
        self.train_output = Y
        #输入层的结点数目
        self.input_n = layer[0]
        #隐含层的结点数目
        self.hide_n = layer[1]
        #输出层的结点数目
        self.output_n = layer[2]
        #训练次数s's's's's's's
        self.num=num
        
        #初始化权值
        self.W1 = 2 * np.random.random((self.hide_n, self.input_n)) - 1
        self.W2 = 2 * np.random.random((self.output_n, self.hide_n)) - 1
        self.max_data = max(self.train_output)
        self.min_data = min(self.train_output)
        #初始化bias
        self.b1 = np.random.rand(self.hide_n, 1) * 2 - 1
        self.b2 = np.random.rand(self.output_n, 1) * 2 - 1
        #类似于中心化
        #self.train_output = (self.train_output -
        #                     self.min_data) / (self.max_data - self.min_data)

        

    def train(self, rate):
        for n in range(self.num):
            for i in range(self.sample_num):
                input_data = self.train_input[i]
                output_data = self.train_output[i]
                #将每一行的输入值分成n个只有一个参数的矩阵
                input_data = input_data.reshape(self.input_n, 1)
                output_data = output_data.reshape(self.output_n, 1)
                
                #神经网络前向计算
                #1.计算隐含层的激活值
                z1 = np.dot(self.W1, input_data) + self.b1
                a1 = relu(z1)
                #2.计算隐含层与输出层之间的激活值
                a1 = a1.reshape(self.hide_n, 1)
                z2 = np.dot(self.W2, a1) + self.b2
                #3.输出层中的激活值
                a2 = relu(z2)
                
                #神经网络反向传播计算
                #损失函数
                loss = (1/2)*((output_data - a2)**2)
                adjustment2 = (output_data-a2) * relu_derivative(a2)
                adjustment1 = relu_derivative(a1) * np.dot(self.W2.T, adjustment2)
                #利用python的广播
                self.W1 += rate * adjustment1
                self.W2 += rate * adjustment2
                
                self.b1 += rate * adjustment1
                self.b2 += rate * adjustment2
        print("训练完成")
        print("损失值为：",loss[0][0])
        
    #输入数据进行预测
    def predit(self, predit_data, rate):
        predit_data = predit_data.reshape(self.input_n, 1)
        self.train(rate)
        z1 = np.dot(self.W1, predit_data) + self.b1
        a1 = relu(z1)
        a1 = a1.reshape(self.hide_n, 1)
        z2 = np.dot(self.W2, a1) + self.b2
        a2 = relu(z2)
        #return (final * (self.max_data - self.min_data) + self.min_data)
        return(a2)

network = BP(X, Y, layer)
predit_data = np.array([3.0,9.3,3.3,2.05,100,2.8,11.2,50])
y = network.predit(predit_data, alpha)
print("预测值为：",y[0][0])
        
            