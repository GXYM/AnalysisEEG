import numpy as np
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    path = "result"
    fn_list = os.listdir(path)
    print(fn_list)
    fn_dat = list()
    for fn in fn_list:
        fn = os.path.join(path, fn)
        with open(fn, 'r', encoding='utf-8') as f:
            datas = f.read().split("\n\n")
            fn_dat.append(datas[:9])

    SVCs = list()
    DTrees = list()
    RForests = list()
    ETrees = list()
    for idx, dat in enumerate(fn_dat):
        SVC = list()
        DTree = list()
        RForest = list()
        ETree = list()
        for dt in dat:
            dl = dt.split("\n")
            SVC.append(float(dl[2]))
            DTree.append(float(dl[5]))
            RForest.append(float(dl[8]))
            ETree.append(float(dl[11]))
        SVCs.append(np.array(SVC)[::-1])
        DTrees.append(np.array(DTree)[::-1])
        RForests.append(np.array(RForest)[::-1])
        ETrees.append(np.array(ETree)[::-1])

    SVCs = np.array(SVCs)
    DTrees = np.array(DTrees)
    RForests = np.array(RForests)
    ETrees = np.array(ETrees)

    fig = plt.figure(figsize=(15, 10))

    for idx, svc in enumerate(SVCs):
        x = range(1, SVCs.shape[1] + 1)
        ax = fig.add_subplot(2, 2, idx + 1)
        ax.plot(x, SVCs[idx, :]*100, label='SVC', color="skyblue", marker="^", ms=6)
        ax.plot(x, DTrees[idx]*100, label=' Decision Tree', color="limegreen", marker="^", ms=6)
        ax.plot(x, RForests[idx]*100, label='Random Forest', color="orange", marker="^", ms=6)
        ax.plot(x, ETrees[idx]*100, label='Extra Tree', color="red", marker="^", ms=6)
        ax.set_title("{}".format(idx + 1), y=0.88, x=0.05)
        ax.plot(1, 40, color="white")
        ax.plot(1, 75, color="white")
        ax.set_xlabel('train copies ', fontsize=10)
        ax.set_ylabel('accuracy (%)', fontsize=10)

        plt.legend(loc=(0.725, 0.02), numpoints=1)
        leg = plt.gca().get_legend()
        ltext = leg.get_texts()
        plt.setp(ltext, fontsize='small')
    # plt.savefig('C4-0.png')

    ax = fig.add_subplot(2, 2,  4)
    ax.plot(x, np.mean(SVCs* 100, axis=0), label='SVC', color="skyblue", linestyle=':',marker="x", ms=6)
    ax.plot(x, np.mean(DTrees * 100,axis=0), label=' Decision Tree', color="limegreen", linestyle=':',marker="x", ms=6)
    ax.plot(x, np.mean(RForests * 100,axis=0), label='Random Forest', color="orange", linestyle=':',marker="x", ms=6)
    ax.plot(x, np.mean(ETrees*100,axis=0), label='Extra Tree', color="red", linestyle=':',marker="x", ms=6)
    ax.set_title("{}".format(4), y=0.88, x=0.05)
    ax.plot(1, 40, color="white")
    ax.plot(1, 75, color="white")
    ax.set_xlabel('train copies ', fontsize=10)
    ax.set_ylabel('accuracy (%)', fontsize=10)

    plt.legend(loc=(0.725, 0.02), numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize='small')
    plt.savefig('C4-2.png')
    plt.show()

