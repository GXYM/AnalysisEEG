import numpy as np
import xlrd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    path="./S4/S4_test_data.xlsx"
    name = path.split("/")[1]
    workbook = xlrd.open_workbook(path)
    sheet_names = workbook.sheet_names()
    print(sheet_names)

    for i, sht_name in enumerate(sheet_names):
        sheet = workbook.sheet_by_name(sht_name)
        print((sheet.name,sheet.nrows,sheet.ncols))

        fig = plt.figure(figsize=(20, 16))
        for idx in range(sheet.ncols):
            cols = sheet.col_values(idx)  # 获取第idx列内容
            ax = fig.add_subplot(5, 4, idx+1)
            ax.plot(range(len(cols)), cols)
            ax.set_title("{}-{}".format(sheet.name, idx))
            # ax.set_xlabel("time")
            ax.set_ylabel("electric potential")
        plt.savefig('./{}/view/{}.png'.format(name, sheet.name))
        # plt.show()
        plt.clf()


