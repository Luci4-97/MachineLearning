import numpy as np
import matplotlib.pyplot as plt
import math


class dbscan(object):
    def __init__(self, use_logo=False):
        if use_logo:
            print('''
            ██████╗ ██████╗ ███████╗ ██████╗ █████╗ ███╗   ██╗
            ██╔══██╗██╔══██╗██╔════╝██╔════╝██╔══██╗████╗  ██║
            ██║  ██║██████╔╝███████╗██║     ███████║██╔██╗ ██║
            ██║  ██║██╔══██╗╚════██║██║     ██╔══██║██║╚██╗██║
            ██████╔╝██████╔╝███████║╚██████╗██║  ██║██║ ╚████║
            ╚═════╝ ╚═════╝ ╚══════╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═══╝

            ''')
        self.__UNCLASSIFIED = False
        self.__NOISE = 0
        pass

    def feed(self, X):
        self.X = X
        if self.X.shape[0] == 0:
            print("[Error] Can't load X_csv.")
        else:
            return 0
        pass

    def cluster(self, eps, minPts):
        """
        输入：数据集, 半径大小, 最小点个数
        输出：分类簇id
        """
        data = np.mat(self.X).transpose()
        clusterId = 1
        nPoints = data.shape[1]
        clusterResult = [self.__UNCLASSIFIED] * nPoints
        for pointId in range(nPoints):
            # point = self.X[:, pointId]
            if clusterResult[pointId] == self.__UNCLASSIFIED:
                if self.__expand_cluster__(data, clusterResult, pointId, clusterId, eps, minPts):
                    clusterId = clusterId + 1
        print("cluster Numbers = ", clusterId - 1)
        self.__plotFeature__(data, clusterResult, clusterId - 1)
        return clusterResult, clusterId - 1

    def __expand_cluster__(self, data, clusterResult, pointId, clusterId, eps, minPts):
        """
        输入：数据集, 分类结果, 待分类点id, 簇id, 半径大小, 最小点个数
        输出：能否成功分类
        """
        seeds = self.__region_query__(data, pointId, eps)
        if len(seeds) < minPts:  # 不满足minPts条件的为噪声点
            clusterResult[pointId] = self.__NOISE
            return False
        else:
            clusterResult[pointId] = clusterId  # 划分到该簇
            for seedId in seeds:
                clusterResult[seedId] = clusterId

            while len(seeds) > 0:  # 持续扩张
                currentPoint = seeds[0]
                queryResults = self.__region_query__(data, currentPoint, eps)
                if len(queryResults) >= minPts:
                    for i in range(len(queryResults)):
                        resultPoint = queryResults[i]
                        if clusterResult[resultPoint] == self.__UNCLASSIFIED:
                            seeds.append(resultPoint)
                            clusterResult[resultPoint] = clusterId
                        elif clusterResult[resultPoint] == self.__NOISE:
                            clusterResult[resultPoint] = clusterId
                seeds = seeds[1:]
            return True

    def __region_query__(self, data, pointId, eps):
        """
        输入：数据集, 查询点id, 半径大小
        输出：在eps范围内的点的id
        """
        nPoints = data.shape[1]
        seeds = []
        for i in range(nPoints):
            if self.__eps_neighbor__(data[:, pointId], data[:, i], eps):
                seeds.append(i)
        return seeds

    def __eps_neighbor__(self, a, b, eps):
        """
        输入：向量A, 向量B
        输出：是否在eps范围内
        """
        return self.__dist__(a, b) < eps

    def __dist__(self, a, b):
        """
        输入：向量A, 向量B
        输出：两个向量的欧式距离
        """
        return math.sqrt(np.power(a - b, 2).sum())

    def __plotFeature__(self, data, clusters, clusterNum):
        matClusters = np.mat(clusters).transpose()
        fig = plt.figure()
        scatterColors = ['black', 'blue', 'green', 'yellow', 'red', 'purple', 'orange', 'brown']
        ax = fig.add_subplot(111)
        for i in range(clusterNum + 1):
            colorSytle = scatterColors[i % len(scatterColors)]
            subCluster = data[:, np.nonzero(matClusters[:, 0].A == i)]
            ax.scatter(subCluster[0, :].flatten().A[0], subCluster[1, :].flatten().A[0], c=colorSytle, marker='.')
        plt.show()
