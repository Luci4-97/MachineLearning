import numpy as np
import time
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt


def initialize(X,y):
    # 初始化数据
    data = X
    answer = y
    cent = data[:3, :]
    return [data, answer, cent]


def cluster(data, centers):
    # 聚类
    dist = []
    for i in range(3):
        delta = data - np.tile(centers[i, :], [data.shape[0], 1])
        delta = delta ** 2
        delta = delta.sum(axis=1)
        dist.append(delta)
    dist = np.asarray(dist)
    flag = []
    for i in range(data.shape[0]):
        tmp = np.array(dist[:, i])
        flag.append(np.where(tmp == tmp.min())[0])
    flag = np.array(flag)
    return flag


def center(data, label, K):
    centers = []
    dat_sorted = []
    mindist = float("inf")
    for k in range(K):
        dat_sorted.append(data[np.where(label == k)[0], :])
    for cluster in dat_sorted:
        tmp_cent = []
        for x in cluster:
            dist = ((cluster - np.tile(x, [cluster.shape[0], 1])) ** 2).sum(axis=0).sum()
            if dist < mindist:
                tmp_cent = x
                mindist = dist
        centers.append(tmp_cent)
        mindist = float("inf")
    centers = np.array(centers)
    return centers


def convergent(pre_cent, cur_cent, epsilon):
    delta = cur_cent - pre_cent
    ret = (delta ** 2).sum(axis=0).sum()
    if ret < epsilon:
        ret = True
    else:
        ret = False
    return ret


def kmediods(X,y):
    # Initialize
    data, answer, cent = initialize(X,y)
    label = cluster(data, cent)
    start_time = time.time()
    epsilon = 0.01
    while True:
        pre_cent = cent
        cur_cent = center(data, label, K=3)
        if convergent(pre_cent, cur_cent, epsilon):
            break
        end_time = time.time()
        if end_time - start_time > 30:
            break
        label = cluster(data, cur_cent)
    label.shape = answer.shape
    lda = LinearDiscriminantAnalysis(n_components=2)
    lda.fit(data, answer)
    x = lda.transform(data)
    ct = lda.transform(cur_cent)
    plt.subplot(121)
    plt.scatter(x[:, 0], x[:, 1], marker='.', c=label)
    plt.scatter(ct[:, 0], ct[:, 1], marker='o', c='r')
    plt.subplot(122)
    plt.scatter(x[:, 0], x[:, 1], marker='.', c=answer)
    plt.show()