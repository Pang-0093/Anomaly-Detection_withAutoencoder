####本文件用于寻找最优误差阈值，并给出相应的误报率
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
import globalvar as gl
import torch
import os
from myAE import model,train_loader
path="C:\\Users\\lenovo\\Desktop\\rush\\AE-Anomaly-Detection"
res=path+"\\results"
#获取用到的变量值
recon_err_test = gl.get_value('recon_err_test')
y_test = gl.get_value('y_test')
X_test = gl.get_value('X_test')
#试验从0到10之间的500个值为阈值
threshold = np.linspace(0, 10, 500)#linspace函数和arrange作用差不多
acc_list = []
f1_list = []
for t in threshold:
    y_pred = (recon_err_test > t).astype(int)#astype函数用于转化dateframe某一列的数据类型
    acc_list.append(accuracy_score(y_pred, y_test))#准确率 判定正确的比例
    f1_list.append(f1_score(y_pred, y_test))#f1score指标 详见笔记


#将各阈值对应的上述指标绘图，从图中选取最高点即为最合适的阈值
plt.figure(figsize=(8, 6))
plt.plot(threshold, acc_list, c='y', label='acc')
plt.plot(threshold, f1_list, c='b', label='f1')
plt.xlabel('threshold')
plt.ylabel('classification score')
plt.legend()
#plt.show()  注意在 plt.show() 后实际上已经创建了一个新的空白的图片（坐标轴），这时候后面的 plt.savefig() 就会保存这个新生成的空白图片。
plt.savefig(res+'\\'+'classification score changes with  thresholds.jpg')

#输出最合适的阈值以及对应的指标
i = np.argmax(f1_list)
t = threshold[i]
score = f1_list[i]
print('Recommended threshold: %.3f, related f1 score: %.3f' % (t, score))

#输出误报率
y_pred = (recon_err_test > t).astype(int)
FN = ((y_test == 1) & (y_pred == 0)).sum()
FP = ((y_test == 0) & (y_pred == 1)).sum()
print('In %d data of test set, FN: %d, FP: %d' % (len(y_test), FN, FP))#输出误报率 FN；假反例  FP：假正例

#利用上面学习好的误差阈值进行精确到每个样本点的异常检测
def reconstructed_error(X):
    return torch.mean((model(X)-X)**2,dim = 1).detach().numpy()
model= torch.load('model.pkl')
plt.figure()
error_array=recon_err_test.tolist()#tolist()将ndarray转换为list 多用debug的变量窗口！机器学习的代码各种变量类型tensor ndarray 再加上python的list很容易搞乱
for i in range(len(error_array)):
    if error_array[i] <= t:
        plt.plot(i,error_array[i],'o',color='blue')
    else:
        plt.plot(i,error_array[i], 'o', color='red')
plt.savefig(res+'\\'+'accurate anomaly detection.jpg')
