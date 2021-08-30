####主成分分析
import pandas as pd
import numpy as np
from scipy.stats import f
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class myPCA:
    #1、类的构造函数，用于初始化类成员等，只能用__init__ ，不能换其他名字
    #2、在Python的类中有很多方法的名字有特殊的重要意义。__init__方法的意义：
    #__init__方法在类的一个对象被建立时，马上运行。这个方法可以用来对你的对象做一些你希望的初始化。
    def __init__(self) -> None:
        self.scaler_ = StandardScaler()  #初始化scaler用于归一化
        self.components_= None  #负载矩阵
        self.n_components_=0    #主元个数
        self.explained_variance_ = None #PCA后的方差（前n_components个特征值）
        self.explained_variance_ratio_ = .00 #方差贡献率
        self.lambdas_= None     # PCA前全部的特征值（奇异值）
        self.data_train_pca_ = None  #训练集PCA后
        self.data_te_pca_ = None     #测试集PCA后
        self.train_T2_ = None    #训练集的T2统计量
        self.te_T2_ = None     #测试集的T2统计量
        self.train_SPE_ = None   #训练集的SPE统计量
        self.te_SPE_ = None    #测试集的SPE统计量
        self.T2_limit_ = .00    #根据训练集计算的T2控制限
        self.SPE_limit_ = .00   #根据测试集计算的SPE控制限
        pass
    # 读入.csv文件转化为numpy矩阵
    def raw_data(self,file):
        data = pd.read_csv(file)
        data = np.array(data)
        data = np.delete(data,0,1)
        return data
    # 根据pca降维后得到的数据集计算T2统计量
    def hotelling_T2(self, train_pca, lambdas):
        T2 = np.array([xi.dot(np.diag(lambdas**-1)).dot(xi.T) for xi in train_pca])
        return T2
    # T2控制限计算
    def T2_limit(self, conf,n,k):
        maxT2 = k * (n**2 - 1)  / (n - k) /n *f.ppf(conf, k, n - k)
        return maxT2
    # 计算SPE（给定PCA前的数据集和负载矩阵p）
    def SPE(self, data,p):
        Q=[]
        n=data.shape[0] # 数据集行数
        m=data.shape[1] # 数据集列数
        for i in range(n):
            temp=np.matmul(data[i],(np.identity(data.shape[1])-p.dot(p.T)))
            temp=np.matmul(temp,data[i].T)
            Q.append(temp)
        return Q
    # SPE控制限theta计算  （算SPE控制限会用到）
    def theta_calculation(self, corr_var, n_components, i, m):
        theta = 0
        for k in range(n_components,m):
            theta += (corr_var[k]**i)
        return theta
    # SPE控制限计算
    def SPE_limit(self, p, eigen, n_comp, n_sensors):
        theta1 = self.theta_calculation(eigen, n_comp, 1, n_sensors)
        theta2 = self.theta_calculation(eigen, n_comp, 2, n_sensors)
        theta3 = self.theta_calculation(eigen, n_comp, 3, n_sensors)
        h0 = 1-((2*theta1*theta3)/(3*(theta2**2)))
        c_alpha = norm.ppf(p)
        limit = theta1*((((c_alpha*np.sqrt(2*theta2*(h0**2)))/theta1)+1+(theta2*h0*(h0-1))/(theta1**2))**(1/h0))
        return limit
    # 得到T2统计量后进行plot
    def plot_T2(self, T2 , T2_limit , conf,plt):
        plt.plot(range(len(T2)), T2, linewidth=2,c='blue')#加上了linewidth格式控制就自动变为绘制线图
        # for i in range(len(T2)):
        #     if T2[i]<=T2_limit:
        #         plt.plot(i,T2[i],'o',color='blue')
        #     else:
        #         plt.plot(i, T2[i], 'o', color='red')
        plt.axhline(y = T2_limit, color='red', ls='--')
        plt.xlabel('Samples')
        plt.ylabel('Hotellings T²')
        plt.title('Hotelling T² Statistics Plot- {} % Confidence level'.format(conf*100))
        plt.grid()
        return 0
    # 得到SPE统计量后进行plot
    def plot_SPE(self, SPE , SPE_limit , conf, plt):
        plt.plot(range(len(SPE)), SPE,  linewidth=2,c='blue')
        # for i in range(len(SPE)):
        #     if SPE[i]<=SPE_limit:
        #         plt.plot(i,SPE[i],'o',color='blue')
        #     else:
        #         plt.plot(i, SPE[i], 'o', color='red')
        plt.axhline(y = SPE_limit, color='red', ls='--')
        plt.xlabel('Samples')
        plt.ylabel('Square Prediction Error')
        plt.title('Square Prediction Error Plot - {} % Confidence level'.format(conf*100))
        plt.grid()
        return 0
    # 指定训练集路径和方差贡献率进行PCA训练
    def AD_PCA_train(self, file_train, contri,conf,pretrain):
         # *************训练集raw data*************
        data_train = self.raw_data(file_train)
        n = np.shape(data_train)[0] # 保存训练集样本数
        m = np.shape(data_train)[1] # 保存样本在原样本空间特征维度
        # *************初始化scalar对训练集和测试集进行归一化操作*************
        if(pretrain==True):
             scaled_train_data = self.scaler_.fit_transform(data_train)
        else:
             scaled_train_data=data_train
        #*************使用PCA基于SVD对训练集进行降维训练*************
        pca=PCA(n_components=contri)    # 降维后的数据保持contri的信息
        self.data_train_pca_=pca.fit_transform(scaled_train_data) # 得到训练集降维后n*k的主成分矩阵

       #记录训练结果：负载矩阵、主成分个数、特征值、特征向量、方差贡献率等，都由PCA包现有函数直接提供
        self.components_=pca.components_.T # 负载矩阵m*k 即负载向量按列向量形式存储
        self.n_components_=np.shape(self.components_)[1] # 主成分个数k
        self.explained_variance_ = pca.explained_variance_ # 特征值向量（从大到小排列共k个）
        self.explained_variance_ratio_ = pca.explained_variance_ratio_  # 方差贡献率
        self.cumulative_variance_contribution_ratio=sum(self.explained_variance_ratio_)#累计方差贡献率

        # 计算得到所有特征值（降维之前的所有特征值）以便后续的SPE控制限计算
        pca2=PCA(n_components=m)#m就是原始样本的维度数 即实际上不发生降维，只是利用了函数里的SVD方法
        pca2.fit_transform(scaled_train_data)
        self.lambdas_ = pca2.explained_variance_#即降维之前的所有的特征值

        #输出信息
        print('The number of principal components:',self.n_components_)
        print('The cumulative variance contribution rate:', self.cumulative_variance_contribution_ratio)
        #np.set_printoptions(threshold=np.inf)#不允许python省略输出
        #print('principal component vectors:',self.components_.T)

    # *************训练集T2统计量*************
        self.train_T2_=self.hotelling_T2(self.data_train_pca_,self.explained_variance_)
        # *************T2控制限计算*************
        self.T2_limit_ = self.T2_limit(conf, n, self.n_components_)
        print('Maximum Confident Limit for T²: {:.2f} ({:.2f} %Confidence)'.format(self.T2_limit_, conf*100))
        # *************训练集SPE统计量计算************
        self.train_SPE_=self.SPE(scaled_train_data,self.components_)
        # *************SPE控制限计算************
        self.SPE_limit_=self.SPE_limit(conf, self.lambdas_, self.n_components_, m)
        print('Maximum Confident Limit for SPE: {:.2f} ({:.2f} %Confidence)'.format(self.SPE_limit_, conf*100))
        print('\n')
        print('\n')
        return 0
    # 用训练后的模型来处理测试集数据，包括：降维求得分矩阵，求T2 SPE统计量
    def AD_PCA_trans(self, file_te):
        data_te = self.raw_data(file_te)
        #scaled_test_data=self.scaler_.transform(data_te)   # 测试集数据归一化
        scaled_test_data=data_te
        temp = pd.DataFrame(scaled_test_data)  # 将标准化后的训练集输出成csv供后面和重构值进行比对（因为重构值是经过标准化的）
        temp.to_csv("C:\\Users\\lenovo\\Desktop\\rush\\AE-Anomaly-Detection\\results\\standard_d02_PCA.csv")
        self.data_te_pca_=np.matmul(scaled_test_data,self.components_) # 测试集数据PCA
        # *************测试集T2统计量*************
        self.te_T2_=self.hotelling_T2(self.data_te_pca_,self.explained_variance_)
        # *************测试集SPE统计量计算************
        self.te_SPE_=self.SPE(scaled_test_data,self.components_)
        return 0

