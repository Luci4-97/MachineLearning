import Kmeans
import Kmediods
import DBSCAN
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import sklearn.cluster as sk
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # 从sklearn导入数据IRIS数据
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # K 中心聚类
    Kmediods.kmediods(X, y)
    # K 均值聚类
    Kmeans.kmeans(X, y)
    # DBSCAN 密度聚类
    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)
    X = lda.transform(X)
    clu = DBSCAN.dbscan()
    clu.feed(X)
    clu.cluster(0.65, 10)

    # sklearn Kmeans
    lda = LinearDiscriminantAnalysis(n_components=2)
    kmeans = sk.KMeans(n_clusters=3)
    y_pre = kmeans.fit_predict(X)
    lda.fit(X, y)
    X = lda.transform(X)
    color = y_pre.astype(np.str)
    color[np.where(y_pre == 0)[0]] = 'c'
    color[np.where(y_pre == 1)[0]] = 'b'
    color[np.where(y_pre == 2)[0]] = 'm'
    plt.scatter(X[:, 0], X[:, 1], c=color, marker='.')
    plt.show()
