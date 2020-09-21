import os
import xlrd
import numpy as np
import torch
import matplotlib.pyplot as plt


class TextSleep(object):
    def __init__(self, data_dir, is_training=True):
        self.is_training = is_training
        self.data_pth = os.path.join(data_dir, "sleepdata.xlsx")

        self.data_workbook = xlrd.open_workbook(self.data_pth)
        self.data_sheet_names = self.data_workbook.sheet_names()

        self.train_data = list()
        self.val_data = list()
        self.test_data = list()
        self.R = 1
        # The corresponding data of each type is divided into 10,
        # and then 2 copies are taken as test set
        # and 1 as verification set.
        # The training set was selected from the remaining 7 copies
        for sht_name in self.data_sheet_names:
            sheet = self.data_workbook.sheet_by_name(sht_name)
            cls_list = list()
            for idx in range(1, sheet.nrows):
                dat = sheet.row_values(idx)  # 获取第idx列内容
                cls_list.append(np.array(dat[:5]))
            k = len(cls_list)
            cls_np = np.stack(cls_list, axis=0)
            np.random.shuffle(cls_np)
            self.train_data.append(cls_np[:k*self.R//10, :])  # get first R copies as train data
            self.val_data.append(cls_np[k*7//10:k*8//10, :])  # get first 1 copies as val data
            self.test_data.append(cls_np[k*8//10:, :])        # get first 2 copies as test data
        self.train_data = np.concatenate(self.train_data, axis=0)
        self.val_data = np.concatenate(self.val_data, axis=0)
        self.test_data = np.concatenate(self.test_data, axis=0)

        self.mean_value = np.mean(self.train_data[:, :], axis=0)
        self.std_value = np.std(self.train_data[:, :], axis=0)
        self.mean_value[0] = 2.0  # label - 2.0
        self.std_value[0] = 1.0   # labels needn't to be  normalization

        self.train_data = (self.train_data - self.mean_value) / self.std_value[0]
        self.val_data = (self.val_data - self.mean_value) / self.std_value[0]
        self.test_data = (self.test_data - self.mean_value)/self.std_value[0]

    def __getitem__(self, item):
        train_data = self.train_data[item]
        train_data = torch.from_numpy(train_data).float()
        return train_data

    def __len__(self):
        return len(self.train_data)


if __name__ == '__main__':

    train_loader = TextSleep(".")
    for i, main_data in enumerate(train_loader):
        print(main_data)





