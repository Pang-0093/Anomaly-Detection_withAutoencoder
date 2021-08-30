#####自编码器
import seaborn as sns #
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
import numpy as np  # linear algebra
import pandas as pd  # read and wrangle dataframes
import matplotlib.pyplot as plt # visualization
import globalvar as gl
import sys
from sklearn.base import TransformerMixin # To create new classes for transformations
from sklearn.preprocessing import (FunctionTransformer, StandardScaler)
from sklearn.model_selection import train_test_split
from collections import Counter
import scipy.io as io
dn="d02_te"
###########结果存储地址
path="C:\\Users\\lenovo\\Desktop\\rush\\AE-Anomaly-Detection"
res=path+"\\results"

############读入数据
X_train = pd.read_csv(path+"\\data"+'\\d00.csv')#读入的数据为DataFrame数据框类型（二维表数据结构，由行列数据组成的表格）
X_train=np.array(X_train)#化为np array数据类型
X_train = np.delete(X_train,0,1)#参数：数据，序号，1为列0为行
X_test=pd.read_csv(path+"\\data"+'\\'+dn+'.csv')
X_test=np.array(X_test)
X_test= np.delete(X_test,0,1)
y_test = np.concatenate([np.zeros(160),np.ones(800)])#y_test是测试集的标记（以160为分界 0表示正常1表示异常）用于测试误报率，训练过程没用到
############数据预处理
sc = StandardScaler()#实例化一个进行数据标准化的类；z = (x - u) / s；且记录下训练集的均值和方差以便之后标准化测试集
X_train = sc.fit_transform(X_train)#标准化训练集
temp = pd.DataFrame(X_train)#将标准化后的训练集输出成csv供后面和重构值进行比对（因为重构值是经过标准化的）
temp.to_csv("C:\\Users\\lenovo\\Desktop\\rush\\AE-Anomaly-Detection\\results\\standard_d00.csv")
X_test = sc.transform(X_test)#标准化测试集
X_train,X_test = torch.FloatTensor(X_train),torch.FloatTensor(X_test)#转换成张量tensor
train_set = TensorDataset(X_train)
# Dataset是一个包装类，用来将数据包装为Dataset类，然后传入DataLoader中。
# DataLoader是一个比较重要的类，它提供的常用操作有：batch_size(每个batch的大小), shuffle(是否进行shuffle操作),
#它有自己的数据类型
train_loader = DataLoader(dataset=train_set, batch_size=10, shuffle=True)#一批数据有10个

############搭建模型
input_size = X_train.size(1)#输入特征个数 本场景下为52个
model = nn.Sequential(#序贯法搭建模型 model是自定义的名称 模型搭建还有其他方法详见笔记
    #压缩部分
    nn.Linear(input_size, 45),#第一层提取45个特征（45个神经元）nn.Linear完成从输入层到隐藏层的线性变换
    nn.Tanh(),#激励函数 奇函数，严格单调递增，限制在两水平渐近线y=1和y=-1之间
    nn.Linear(45, 40),
    nn.Tanh(),
    nn.Linear(40, 35),
    nn.Tanh(),
    nn.Linear(35, 30),#最后提取到30个特征。选取30依据的是之前PCA分析出的主成分个数
    nn.Tanh(),
    #后面是重构部分
    nn.Linear(30, 35),
    nn.Tanh(),
    nn.Linear(35, 40),
    nn.Tanh(),
    nn.Linear(40, 45),
    nn.Tanh(),
    nn.Linear(45, input_size)
)
num_epochs = 1000#进行1000代训练
optimizer = torch.optim.Adam(model.parameters(), 0.001)#构造优化器 Adam(Adaptive Moment Estimation)本质上是带有动量项的RMSprop，
# 它利用梯度的一阶矩估计和二阶矩估计动态调整每个参数的学习率。它的优点主要在于经过偏置校正后，每一次迭代学习率都有个确定范围，使得参数比较平稳。
loss_func = nn.MSELoss()#定义损失函数为均方误差，参与运算的是两个同维度向量，对应位置两元素求差的方，构成的均方误差也是一个同维度向量


#######开始训练
for epoch in range(num_epochs):
    total_loss = 0.
    for step, (x,) in enumerate(train_loader):#train_loader里面一个元素就是一个batch
        #enumerate() 对某数据类型内的元素遍历，并返回一个元组：序号，元素值
        x_reconstructed = model(x)
        loss = loss_func(x_reconstructed, x)
        optimizer.zero_grad() #梯度清零
        loss.backward()#反向传播求梯度
        optimizer.step()#迭代更新
        total_loss += loss.item() * len(x) #.item()方法 是得到一个元素张量里面的元素值，具体就是用于将一个零维张量转换成浮点数，比如计算loss，accuracy的值
                                           #×一个本片数据的个数，将来再÷整个数据集的个数，相当于做了个加权平均
    total_loss /= len(train_set)
    print('Epoch {}/{} : loss: {:.4f}'.format(epoch + 1, num_epochs, total_loss))


######保存模型
torch.save(model, 'model.pkl')  #将训练好的模型存储下来，存储模型有几种模式：全部、只存参数等；此处为全部
torch.save(model.state_dict(), 'model_params.pkl')


########计算重构误差，绘制误差分布密度函数
def reconstructed_error(X):
    return torch.mean((model(X)-X)**2,dim = 1).detach().numpy()
recon_err_train = reconstructed_error(X_train)
recon_err_test = reconstructed_error(X_test)
recon_err = np.concatenate([recon_err_train,recon_err_test])
labels = np.concatenate([np.zeros(len(recon_err_train)),y_test])#区分正常与异常数据以分别绘制其误差分布密度函数
#sns.set_style(style='darkgrid')#底色控制函数
sns.kdeplot(recon_err[labels==0], label="d02")#绘制核密度估计，用来绘制特征变量y值的分布
sns.kdeplot(recon_err[labels==1], label=dn)
plt.legend()#显示matplotlib的图例必须要有：1.绘图语句里有label控制项 2.有.legend函数
plt.savefig(res+'\\'+'MSE of d00 and'+dn+'.jpg')




#############赋值全局变量供测试文件调用
gl._init()
gl.set_value('recon_err_test',recon_err_test )
gl.set_value('y_test',y_test )
gl.set_value('score', 90)
gl.set_value('X_test', X_test)
gl.set_value('X_train',X_train)



