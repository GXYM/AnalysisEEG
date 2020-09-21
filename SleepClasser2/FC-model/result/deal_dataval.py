import numpy as np
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    path = "./dataval"
    fn_list = os.listdir(path)
    print(fn_list)
    fn_dat = list()
    for fn in fn_list:
        fn = os.path.join(path, fn)
        with open(fn, 'r') as f:
            datas = f.readlines()
            cov_data = list()
            for dat in datas:
                cov_data.append([float(i) for i in dat.split(",")])
            fn_dat.append(cov_data)

    fig = plt.figure(figsize=(15, 10))
    val_data = list()
    test_data = list()
    for idx, dat in enumerate(fn_dat):
        dat = np.array(dat)
        x = range(1, dat.shape[1]+1)
        index = np.argsort(dat[2, 300:], axis=0)[::-1][0:50]
        test_data.append([np.mean(dat[1, index]), np.max(dat[1, index]), np.min(dat[1, index]), np.std(dat[1, index])])
        val_data.append([np.mean(dat[2, index]), np.max(dat[2, index]), np.min(dat[2, index]), np.std(dat[2, index])])
        ax = fig.add_subplot(3, 3, idx + 1)
        ax.plot(x[::5], dat[0][::5], label='train set', color="skyblue")
        ax.plot(x[::5], dat[1][::5], label='test set', color="limegreen")
        ax.plot(x[::5], dat[2][::5], label='val set', color="orange")
        ax.set_title("{}".format(idx+1), y =0.88 , x=0.05)
        ax.plot(1, 55, color="white")
        ax.plot(1, 100, color="white")
        ax.set_xlabel('epoch ', fontsize=10)
        ax.set_ylabel('accuracy (%)', fontsize=10)

    plt.legend(loc=(0.705, 0.00), numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize='small')

    val_data = np.array(val_data)
    ax = fig.add_subplot(3, 3, 8)
    ax.plot(range(1, val_data.shape[0]+1), val_data[:, 0], label='mean accuracy', color="red", marker="^", ms=6)
    ax.plot(range(1, val_data.shape[0]+1), val_data[:, 1], label='max accuracy', color="limegreen", marker="^", ms=6)
    ax.plot(range(1, val_data.shape[0]+1), val_data[:, 2], label='min accuracy', color="orange", marker="^", ms=6)
    ax.plot(1, 40, color="white")
    ax.plot(1, 80, color="white")
    ax.set_title("{} (val set)".format(8), y=0.88, x=0.17)

    ax.set_xlabel('train copies ', fontsize=10)
    ax.set_ylabel('accuracy (%)', fontsize=10)
    # plt.savefig("{}.png".format(path.split("/")[-1].split(".")[0]))
    plt.legend(loc=(0.58, 0.02), numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize='small')

    test_data = np.array(test_data)
    ax = fig.add_subplot(3, 3, 9)
    ax.plot(range(1, test_data.shape[0] + 1), test_data[:, 0], label='mean accuracy',linestyle=':', color="red", marker="^", ms=6)
    ax.plot(range(1, test_data.shape[0] + 1), test_data[:, 1], label='max accuracy', linestyle=':', color="limegreen", marker="^", ms=6)
    ax.plot(range(1, test_data.shape[0] + 1), test_data[:, 2], label='min accuracy', linestyle=':', color="orange", marker="^", ms=6)
    ax.plot(1, 40, color="white")
    ax.plot(1, 80, color="white")
    ax.set_title("{} (test set)".format(9), y=0.88, x=0.17)

    ax.set_xlabel('train copies ', fontsize=10)
    ax.set_ylabel('accuracy (%)', fontsize=10)
    # plt.savefig("{}.png".format(path.split("/")[-1].split(".")[0]))
    plt.legend(loc=(0.58, 0.02), numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize='small')
    plt.savefig('C4-0.png')
    plt.show()

