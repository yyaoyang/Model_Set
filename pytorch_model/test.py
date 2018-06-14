# -*- encoding=utf-8 -*-

#from sklearn import svm
#from sklearn.externals import joblib
import pandas as pd
import numpy as np
def to_svm(crrc_demo):
    error_code_series = crrc_demo[len(crrc_demo.columns)-1]
    labels=[]
    for error_code_str in error_code_series:
        error_code_list = error_code_str.split(',')
        if '1047' in error_code_list:
            labels.append(1)
        else:
            labels.append(0)
    csv=pd.DataFrame(labels)
    csv.to_csv("./data/error_code_series.csv", index=False)

    for i in range(3):
        del crrc_demo[len(crrc_demo.columns) - 1]
    data=pd.DataFrame(crrc_demo)
    data.to_csv("./data/data_code_series.csv", index=False)
    return crrc_demo, labels

# 用svm二分类
def svm_func(crrc_demo):
    X = pd.read_csv("./data/data_code_series.csv")
    y = pd.read_csv('./data/error_code_series.csv')
    print(y.shape)
    y=y.astype('int')
    clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(X, y)
    joblib.dump(clf, "clf.m")
def pre():
    clf = joblib.load('./data/clf1.m')
    data = pd.read_csv("./data/data_code_series1.csv")

    print(data.shape)
    labels= pd.read_csv("./data/error_code_series.csv")
    y=clf.predict(data)
    l=labels
    y=np.reshape(y,l.shape)
    acc=abs(l-y)
    correct=0
    for i in acc:
        if i==0:
            correct +=1
    acc=correct/l.shape[0]
    print(acc)

if __name__ == "__main__":
    #crrc_demo = pd.read_csv("./data/CRRC2.csv", header=None, encoding='gbk')
    #crrc_demo, labels=to_svm(crrc_demo)
    #svm_func(crrc_demo)
    #pre()
    #h=np.array([[1],[2],[3],[4],[5]])
    co=0
    h=[1,0,1,0,0,0]
    c=[0,0,0,0,0,0]
    co += (int(h == c)).sum()
    print(co)
    #print(np.interp(2.5, xp, fp))

