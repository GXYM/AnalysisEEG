import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    path = "./result/dataval/7.txt"
    with open(path, 'r') as f:
        datas = f.readlines()

    cov_data = list()
    for dat in datas:
        cov_data.append([float(i) for i in dat.split(",")])

    plt.figure(figsize=(8, 5))
    plt.ylim([55, 100])
    axes = plt.subplot()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    axes.set_xlabel('epoch ', fontsize=12)
    axes.set_ylabel('accuracy (%)', fontsize=12)

    x = range(1, len(cov_data[0])+1)
    print(sum(cov_data[1][250:])/(len(cov_data[1][250:])))

    axes.plot(x[::5], cov_data[0][::5], label='train set', color="skyblue", ms=6)
    axes.plot(x[::5], cov_data[1][::5], label='test set', color="limegreen", linewidth=2, ms=6)
    axes.plot(x[::5], cov_data[2][::5], label='val set', color="orange", linewidth=2, ms=4)
    axes.grid(axis="y")
    # axes.spines['bottom'].set_position(('data', 0))
    # axes.spines['left'].set_position(('data', 0))
    # axes.spines['right'].set_position(('data', 0))
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 30,
             }
    #axes.legend(loc="lower right", fontsize=12, bbox_to_anchor=(1.0, 0.0))
    axes.legend()
    plt.savefig("{}.png".format(path.split("/")[-1].split(".")[0]))
    plt.show()


