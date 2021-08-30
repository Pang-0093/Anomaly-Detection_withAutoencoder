####本文件用于利用PCA算法分析重构值特性
from myPCA import myPCA
from myAE import model,dn
import os
import re
import matplotlib.pyplot as plt
import pandas as pd
import globalvar as gl
import torch


contri=0.9
conf=0.95
path="C:\\Users\\lenovo\\Desktop\\rush\\AE-Anomaly-Detection"
res=path+"\\results"
X_test = gl.get_value('X_test')
X_train = gl.get_value('X_train')
if not os.path.exists(res):
    os.mkdir(res)


#用d00作为训练集训练PCA主元模型 （d00是正常数据，AE也是用d00作为训练集）
mypca=myPCA()
file_train=path+'\\data'+'\\d00.csv'
mypca.AD_PCA_train(file_train,contri,conf,True)
 # 保存pca后的结果输出为.csv
k=mypca.n_components_
columns=[]
for i in range(k):
 col='pc'+str(i+1)
 columns.append(col)
temp=pd.DataFrame(mypca.data_train_pca_, columns=columns)#创建工作表用以存储训练集得分矩阵；参数：数据，行名，列名
temp.to_csv(res+'\\'+'d00'+'_components'+'.csv')
# 训练集T2 SPE图
plt.figure(figsize=(20,10), dpi=100)
plt.subplot(211)
mypca.plot_T2(mypca.train_T2_,mypca.T2_limit_,conf,plt)
plt.grid(b=False)#网格线格式控制要放在plot后面！
plt.subplot(212)
mypca.plot_SPE(mypca.train_SPE_,mypca.SPE_limit_,conf,plt)
plt.grid(b=False)
plt.savefig(res+'\\'+'d00.jpg')
plt.close()


#对AE对某测试集如d01_te的重构值用上述主元模型进行分析
model2 = torch.load('model.pkl')
X_reconstructed = model2(X_test)
X_reconstructed_np = X_reconstructed.detach().numpy()
temp = pd.DataFrame(X_reconstructed_np)
temp.to_csv(res + '\\' + 'reconstructed_'+dn+'.csv')
file_test=res+'\\reconstructed_'+dn+'.csv'
#file_test=path+'\\data'+'\\d02_te.csv'
mypca.AD_PCA_trans(file_test)
# 输出测试集重构值进行pca后的结果
k=mypca.n_components_
columns=[]
for i in range(k):
    col='pc'+str(i+1)
    columns.append(col)
temp=pd.DataFrame(mypca.data_te_pca_, columns=columns)#创建工作表用以存储测试集得分矩阵；参数：数据，行名，列名
temp.to_csv(res+'\\'+'reconstructed_'+dn+'_components'+'.csv')

# 测试集重构值的T2
# 测试集重构值的SPE
plt.figure(figsize=(20,10), dpi=100)#dpi分辨率
plt.subplot(211)
mypca.plot_T2(mypca.te_T2_,mypca.T2_limit_,conf,plt)
plt.grid(b=False)#去掉网格线   （网格线格式控制要放在plot后面！ ）
plt.subplot(212)
mypca.plot_SPE(mypca.te_SPE_,mypca.SPE_limit_,conf,plt)
plt.grid(b=False)
plt.savefig(res+'\\'+'reconstructed_'+dn+'.jpg')
plt.close()