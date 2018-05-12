import Kmeans
import Kmediods
import DBSCAN
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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
    clu.cluster(1,3)
