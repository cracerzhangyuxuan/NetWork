import numpy as np
import scipy.special
import matplotlib.pyplot as plt
from PIL import Image


#神经网络为的就是通过训练获取权重、学习率、迭代次数
class neuralNetwork:

    #神经网络初始化
    def __init__(self,inputnodes,hiddennodes,outputnodes,learingrate):
        #输入层节点数
        self.inodes=inputnodes
        #隐藏层节点数
        self.hnodes=hiddennodes
        #输出层节点数
        self.onodes=outputnodes
        #学习率
        self.lr=learingrate


        #初始化输入层与隐藏层之间的权重
        self.wih=np.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        #初始化隐藏层与输出层的权重
        self.who=np.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        #激活函数(激励函数)
        self.activation_function=lambda x: scipy.special.expit(x)

    #数据集训练
    def train(self,inputs_list,targets_list):
        #将输入数据转化成二维矩阵
        inputs=np.array(inputs_list,ndmin=2).T
        #将输出目标转化为二维矩阵
        targets=np.array(targets_list,ndmin=2).T

        #计算隐藏层的输入
        hidden_inputs=np.dot(self.wih,inputs)
        #计算隐藏层的输出
        hidden_outputs=self.activation_function(hidden_inputs)

        #计算输出层的输入
        out_inputs=np.dot(self.who,hidden_outputs)
        #计算输出层的输出
        out_outputs=self.activation_function(out_inputs)

        #计算输出层的误差
        out_errors=targets-out_outputs
        #计算隐藏层误差
        hidden_errors=np.dot(self.who.T,out_errors)

        #更新隐藏层与输出层之间的权重
        self.who+=self.lr*np.dot((out_errors*out_outputs*(1.0-out_outputs)),np.transpose(hidden_outputs))

        #更新输入层与隐藏层之间的权重
        self.wih+=self.lr*np.dot((hidden_errors*hidden_outputs*(1.0-hidden_outputs)),np.transpose(inputs))

    def setweights(self, wih, who):
        self.wih = wih
        self.who = who

    #测试神经网络
    def test(self,inputs_list):
        #将输入数据转化成二维矩阵
        inputs=np.array(inputs_list,ndmin=2).T

        #计算隐藏层的输入
        hidden_inputs=np.dot(self.wih,inputs)
        #计算隐藏层的输出
        hidden_outputs=self.activation_function(hidden_inputs)

        # 计算输出层的输入
        out_inputs = np.dot(self.who, hidden_outputs)
        # 计算输出层的输出
        out_outputs = self.activation_function(out_inputs)

        return out_outputs

if __name__=="__main__":
    #初始化784(28*28)个输入节点,100个隐藏层节点,10个输出节点(0~9)
    input_nodes=784
    hidden_nodes=200
    out_nodes=10


    #学习率
    learning_rate=0.1
    #训练次数
    epochs=5
    #初始化神经网络
    n=neuralNetwork(input_nodes,hidden_nodes,out_nodes,learning_rate)

    #读取训练集
    training_data_file=open('mnist_dataset/mnist_train_100.csv','r')
    training_data_list=training_data_file.readlines()
    training_data_file.close()
    
    
    #训练数据
    for e in range(epochs):
        for record in training_data_list:
            all_values=record.split(',')    #数据用逗号分割
            #输入数据范围(0.01~1)
            inputs=np.asfarray(all_values[1:])/255.0*0.99+0.01  #第一个数据不要，因为是输出值
            #标记数据(相应标记为0.99,其余为0.01)
            targets=np.zeros(out_nodes)+0.01
            targets[int(all_values[0])]=0.99
            n.train(inputs,targets)

    
    #读取测试数据
    test_data_file=open('mnist_dataset/mnist_test_10.csv', 'r')
    test_data_list=test_data_file.readlines()
    test_data_file.close()

    #打印测试数据标签
    for i in range(0,10):
        test_data=test_data_list[i].split(',')
        print('原标签: ',test_data[0])

    #生成标签图片

        image_array=np.asfarray(test_data[1:]).reshape(28,28)   #将测试数据集出第一个数据之后的数据生成28*28的矩阵
        plt.imshow(image_array,cmap='Greys',interpolation='None')
        plt.show()

    #利用神经网络预测

        results=n.test(np.asfarray(test_data[1:])/255.0*0.99+0.01)
        pre_label=np.argmax(results)    #找到最大结果值得索引
        np.set_printoptions(suppress=True)  #不以科学计数法形式展现，以小数形式展现
        print('预测结果: ',pre_label)
        print(results)
    """
    img=Image.open('picture/21.jpg').convert('L')
    newimg=img.resize((28,28),Image.ADAPTIVE)
    test_pic=np.array(newimg)

    wih = np.loadtxt(open('weights/wih_60000.csv'), delimiter=",", skiprows=0)
    who = np.loadtxt(open('weights/who_60000.csv'), delimiter=",", skiprows=0)
    n.setweights(wih, who)
    resluts=n.test(np.asfarray(255.0-test_pic.flatten())/255.0*0.99+0.01)
    np.set_printoptions(suppress=True)
    pre_label=np.argmax(resluts)
    print('预测结果：', pre_label)
    print(resluts)
    """
