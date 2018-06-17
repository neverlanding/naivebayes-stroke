# -*- coding: utf-8 -*-

#哑变量处理
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
#模型评估（度量）
import sklearn.metrics as mt

filename = 'E:/pyworkspace/dataset/test01.tsv'
dataSet = pd.read_csv(filename, sep = '	')
#dataSet = dataSet.fillna(8)

feature=dataSet.ix[:,:-1]
result=dataSet.ix[:,-1]
#result=dataSet.ix[:,-1].astype(str) #对SVM分类列单独处理
prdesNum=50#测试次数
preds=[]#记录测试多次的精度（模型准确度）
Setpreds=[]#记录测试多次的精度（脑卒中精度）
rightpre=[]#检出率（召回率）
F1pre=[]#F1-score（综合脑卒中精度与召回率）
for i in range(prdesNum):
    f_train,f_test,r_train,r_test=train_test_split(feature,result,test_size=.3)
    
#    clf = OneVsRestClassifier(SVC(kernel='linear', probability=True))#SVM
#    clf = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)#多项贝叶斯
#    clf = GaussianNB()#高斯贝叶斯
    clf= RandomForestClassifier(max_depth=7,n_estimators=21)#随机森林   
    clf.fit(f_train, r_train)#训练
    predicted = clf.predict(f_test)  
    preds.append(np.mean(predicted == r_test))
    
    #计算设定概率值的患脑卒中的精度
    r_test=r_test.reset_index(drop=True)
    prePro=pd.DataFrame(clf.predict_proba(f_test),columns=['not','is'])
    selectPro=prePro[prePro['is']>0.555] #测试样本测出为脑卒中的样本
    
    count=0#记录测出为脑卒中样本中实际也为脑卒中的数量
    for i in selectPro.index.values:
        if(r_test.loc[i]==1):
            count=count+1
    Setpreds.append(count/len(selectPro))
    rightpre.append(count/len(r_test[r_test==1]))
    F1pre.append(2*count/len(selectPro)*count/len(r_test[r_test==1])/(count/len(selectPro)+count/len(r_test[r_test==1])))
print(dataSet.columns)
#print(preds)
print('模型准确率为%f' %np.mean(preds))
print('特定概率下脑卒中精度为%f' %np.mean(Setpreds))
print('脑卒中召回率（检出率）为%f' %np.mean(rightpre))
print('脑卒中F1-Score为%f' %np.mean(F1pre))
#print('accuracy_score%f' %mt.accuracy_score(r_test,predicted))
#print('accuracy_score%f' %mt.recall_score(r_test,predicted))
#print('accuracy_score%f' %mt.f1_score(r_test,predicted))
#print('precision_recall_fscore_support:' )
#print(mt.precision_recall_fscore_support(r_test,predicted))



'''
################################################################
绘制箱线图
################################################################
''' 
"""
绘制特征的箱线图
"""
import matplotlib.pyplot as plt
dataSet.boxplot()
plt.show()


'''
################################################################
绘制ROC曲线
################################################################
''' 
'''
"""
绘制脑卒中概率模型的ROC曲线
"""
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


fpr = dict()
tpr = dict()
roc_auc = dict()
#test =np.array(r_test)
test =label_binarize(np.array(r_test), classes=['0', '1'])
fpr[0], tpr[0], _ = roc_curve(test, np.array(predictvalue))
#计算AUC曲线下面积area under curve
roc_auc[0] = auc(fpr[0], tpr[0])


plt.figure()
lw = 2
plt.plot(fpr[0], tpr[0], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('I-特异度')
plt.ylabel('灵敏度')
plt.title('脑卒中概率模型ROC曲线')
plt.legend(loc="lower right")
plt.show()
'''
            