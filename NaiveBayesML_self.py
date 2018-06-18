# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split    

#获取各个类别条件概率
def get_pred(dataSet,inputSimple):
    p0classData = []#初始化类别矩阵
    p1classData = []
    classLabels = dataSet[dataSet.columns[-1]]#选取类别列
    for i in range(len(dataSet.columns) - 1):
        columnLabels = dataSet[dataSet.columns[i]]#特征列
        pData = pd.concat([columnLabels, classLabels], axis = 1)#拼接特征列和类别列
        classSet = list(set(classLabels))
        for pclass in classSet:
            filterClass = pData[pData[pData.columns[-1]] == pclass]#根据类别划分数据集
            filterClass = filterClass[pData.columns[-2]]
            '''判断是离散还是连续类型的特征，分别进行处理 
            if isinstance(inputSimple[i], float):#判断是否是连续变量
                classVar = np.var(filterClass)#方差
                classMean = np.mean(filterClass)#均值
                pro_l = 1/(np.sqrt(2*np.pi) * np.sqrt(classVar))
                pro_r = np.exp(-(inputSimple[i] - classMean)**2/(2 * classVar))
                pro = pro_l * pro_r#概率
                if pclass == '是':
                    p0classData.append(pro)
                else:
                    p1classData.append(pro)
            else:
            '''
            classNum = np.count_nonzero(filterClass == inputSimple[i])#计算属于样本特征的数量
            pro = (classNum + 1)/(len(filterClass) + len(set(filterClass)))#此处进行了拉普拉斯修正
            if pclass == '0':
                p0classData.append(pro)
            else:
                p1classData.append(pro)
    return p0classData, p1classData



if __name__ == '__main__':
    starttime = datetime.datetime.now()
    filename = 'E:/pyworkspace/dataset/stroke/test01.tsv'
    dataSet = pd.read_csv(filename, sep = '	',dtype=object)
#    dataSet = pd.read_csv(filename, sep = '	',dtype={'BMI':np.float,'IS_STROKE':str})
    feature=dataSet.ix[:,:-1]
    result=dataSet.ix[:,-1]
    ave=[]
    actave=[]
    rightpre=[]
    F1pre=[]
    for i in range(3):
        
        f_train,f_test,r_train,r_test=train_test_split(feature,result,test_size=.3)
        trainingSet=pd.concat([f_train, r_train], axis = 1)
        testSet=pd.concat([f_test, r_test], axis = 1).reset_index(drop=True)
        ''' 测试单个输入样本返回概率结果         
        inputSimple = ['1','3','','','4','3','1']
        p0classData, p1classData = get_pred(dataSet, inputSimple)
        if np.prod(p1classData)/(np.prod(p1classData)+np.prod(p0classData)) >0.5:#计算条件概率的累积
            print('患脑卒中的概率为：'+str(np.prod(p1classData)/(np.prod(p1classData)+np.prod(p0classData))))
        else:
            print('患脑卒中的概率为：'+str(np.prod(p1classData)/(np.prod(p-1classData)+np.prod(p0classData))))
        '''           
        testData =[list(testSet.ix[i][:-1]) for i in range(0,len(testSet))]#list化
        testLabels = []
        predictvalue =[] #存储计算ROC需要的判断为脑卒中的数值
        
        for test in testData:
            p0classData, p1classData = get_pred(trainingSet, test)
            predictvalue.append(np.prod(p1classData))
        #    if np.prod(p0classData) > np.prod(p1classData):
            if np.prod(p1classData)/(np.prod(p1classData)+np.prod(p0classData)) >0.5:#计算条件概率的累积
                testLabels.append('1')#保存测试结果
            else:
                testLabels.append('0')
        accuracy = np.mean(testLabels == testSet[testSet.columns[-1]])
        ave.append(accuracy)
              
        # 将测试集正确结果跟预测结果合并到一个DataFrame 
        actresult=pd.DataFrame(list(zip(testSet[testSet.columns[-1]],testLabels)),columns=['act','pre'])
        actpre=len(actresult[(actresult.pre=='1')&(actresult.act=='1')])/len(actresult[actresult.pre=='1'])
        actave.append(actpre)
        rightpre.append(len(actresult[(actresult.pre=='1')&(actresult.act=='1')])/len(r_test[r_test=='1']))
        F1pre.append(2*actpre*len(actresult[(actresult.pre=='1')&(actresult.act=='1')])/len(r_test[r_test=='1'])/(actpre+len(actresult[(actresult.pre=='1')&(actresult.act=='1')])/len(r_test[r_test=='1'])))

    print(dataSet.columns)
    print('模型准确度为%f' %np.mean(ave))
    print('患脑卒中的病人模型精度为%f' %np.mean(actave))
    print('患脑卒中的病人检出率（召回率）为%f' %np.mean(rightpre))
    print('患脑卒中的F1-Scroe为%f' %np.mean(F1pre))
    endtime = datetime.datetime.now()
    print ((endtime - starttime).seconds)
    
'''
################################################################
绘制ROC曲线
################################################################
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
    
            