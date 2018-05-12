import numpy as np
import time
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt


def initialize(X,y):
    # 从sklearn导入数据IRIS数据
    data = X
    answer = y
    cent = data[:3, :]
    return [data, answer, cent]


def cluster(data, means):
    # 聚类
    k = means.shape[0]
    dist = []
    for i in range(3):
        delta = data - np.tile(means[i, :], [data.shape[0], 1])
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


def mean(data, label):
    means = []
    class1 = data[np.where(label == 0)[0], :]
    class2 = data[np.where(label == 1)[0], :]
    class3 = data[np.where(label == 2)[0], :]
    mean1 = class1.sum(axis=0) / class1.shape[0]
    mean2 = class2.sum(axis=0) / class2.shape[0]
    mean3 = class3.sum(axis=0) / class3.shape[0]
    means.append(mean1)
    means.append(mean2)
    means.append(mean3)
    means = np.array(means)
    return means


def convergent(pre_cent, cur_cent, epsilon):
    delta = cur_cent - pre_cent
    ret = (delta ** 2).sum(axis=0).sum()
    if ret < epsilon:
        ret = True
    else:
        ret = False
    return ret


def kmeans(X,y):
    # Initialize
    data, answer, cent = initialize(X,y)
    label = cluster(data, cent)
    start_time = time.time()
    epsilon = 0.01
    while True:
        pre_cent = cent
        cur_cent = mean(data, label)
        if convergent(pre_cent, cur_cent, epsilon):
            break
        end_time = time.time()
        if end_time - start_time > 10:
            break
        label = cluster(data, cur_cent)
    label.shape = answer.shape
    lda = LinearDiscriminantAnalysis(n_components=2)
    lda.fit(data, answer)
    x = lda.transform(data)
    plt.subplot(121)
    plt.scatter(x[:, 0], x[:, 1], marker='.', c=label)
    plt.subplot(122)
    plt.scatter(x[:, 0], x[:, 1], marker='.', c=answer)
    plt.show()