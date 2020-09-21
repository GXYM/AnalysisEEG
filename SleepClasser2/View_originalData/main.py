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

from sklearn import manifold
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
    Color = ["red", "blue", "yellow", "green", "black"]
    X, Y = loadData('sleepdata.xlsx')
    X_embedded = manifold.TSNE(n_components=2).fit_transform(X)
    print("Sklearn TSNE cost time: %s" % str(round(2, 2)))
    figure1 = plt.figure(figsize=(15, 10))
    plt.subplot(1,1,1)
    for i in range(5):
        xxx1 = X_embedded[Y == i, 0]
        xxx2 = X_embedded[Y == i, 1]
        plt.scatter(xxx1, xxx2, c=Color[i])
    plt.xlim(np.min(X_embedded) - 5, np.max(X_embedded) + 5)
    plt.xlim(np.min(X_embedded) - 5, np.max(X_embedded) + 5)
    plt.title('Sklearn TSNE')
    plt.savefig('C4-3.png')
    plt.show()


