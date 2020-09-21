# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 16:38:18 2018

@author: aoanng
"""

import csv
from random import seed
from random import randrange
import numpy as np
import xlrd

from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR  # SVM中的回归算法
from sklearn.svm import SVC
from sklearn import metrics #模型结果评价包
import matplotlib.pyplot as plt


def loadData(filename):#加载数据，一行行的存入列表
    data_workbook = xlrd.open_workbook(filename)
    data_sheet_names = data_workbook.sheet_names()
    all_data = list()
    for sht_name in data_sheet_names:
        sheet = data_workbook.sheet_by_name(sht_name)
        for idx in range(1, sheet.nrows):
            dat = sheet.row_values(idx)  # 获取第idx列内容
            all_data.append(np.array(dat[:5]))

    cls_np = np.stack(all_data, axis=0)

    X = cls_np[:, 1:5].astype(float)

    Y = (cls_np[:, 0]-2).astype(int)

    return X, Y


if __name__ == '__main__':
    seed(1)
    X, Y = loadData('sleepdata.xlsx')
    data_X = preprocessing.StandardScaler().fit_transform(X)

    fw = open("gt_resrult.txt", 'w')

    random_seed = 2020 #2020, 0, 1997

    X_train, X_test, y_train, y_test = train_test_split(data_X, Y, test_size=0.5, random_state=random_seed, shuffle=True)

    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.8, random_state=random_seed,shuffle=True)


    svm = SVC(kernel='rbf', C=30, gamma=0.75, probability=True, max_iter=1000).fit(X_train, y_train)

    tr_p = svm.predict(X_test)
    fw.write("gt,"+ ",".join([str(i) for i in y_test])+"\n")
    fw.write("svm," + ",".join([str(i) for i in tr_p])+"\n")

    print("SVC :")
    print(metrics.r2_score(y_test, tr_p))
    print(metrics.mean_squared_error(y_test, tr_p))


    # ## 决策树
    clf1 = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=random_seed).fit(X_train, y_train)
    tr_p = clf1.predict(X_test)
    print("决策树 :")
    print(metrics.r2_score(y_test, tr_p))
    print(metrics.mean_squared_error(y_test, tr_p))


    ## 随机森林
    clf2 = RandomForestClassifier(n_estimators=100, max_depth=None,
                                  min_samples_split=2, random_state=random_seed).fit(X_train, y_train)
    tr_p = clf2.predict(X_test)
    print("随机森林:")
    print(metrics.r2_score(y_test, tr_p))
    print(metrics.mean_squared_error(y_test, tr_p))

    fw.write("RandomForest," + ",".join([str(i) for i in tr_p])+"\n")


    ## ExtraTree分类器集合
    clf3 = ExtraTreesClassifier(n_estimators=100, max_depth=None,
                                min_samples_split=2, random_state=random_seed).fit(X_train, y_train)
    tr_p = clf3.predict(X_test)
    print("ExtraTree分类器集合:")
    print(metrics.r2_score(y_test, tr_p))
    print(metrics.mean_squared_error(y_test, tr_p))

    fw.write("ExtraTree," + ",".join([str(i) for i in tr_p]) + "\n")
    fw.close()
