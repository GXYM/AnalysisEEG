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



if __name__ == '__main__':

    fc_pred = list()
    with open("FC_result.txt", 'r') as f:
        lines = f.readlines()
        gt_0 = [int(i) for i in lines[0].split(",")[1:]]
        # for d in lines[1:]:
        #     fc_pred.append([int(i) for i in d.split(",")[1:]])

    with open("gt_resrult.txt", 'r') as f:
        lines = f.readlines()
        gt_1 = [int(i) for i in lines[0].split(",")[1:]]
        for d in lines[1:]:
            fc_pred.append([int(i) for i in d.split(",")[1:]])

    gt_0 = np.array(gt_0)
    gt_1 = np.array(gt_1)

    gt = gt_0- gt_1
    print(np.where(gt!=0))

    fc_pred = np.array(fc_pred)
    all_pred = np.sum(fc_pred, axis=0)/fc_pred.shape[0]
    all_pred = np.rint(all_pred)
    print(metrics.r2_score(gt_0, all_pred))
    for pred in fc_pred:
        print(metrics.r2_score(gt_0, pred))



