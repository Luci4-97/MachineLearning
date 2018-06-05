import numpy as np
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.externals import joblib


class import_data(object):
    """
    导入数据类
    """

    def __init__(self):
        print("Loading data...")
        self.X = np.loadtxt("data.csv", delimiter=',')
        self.y = np.loadtxt("label.csv", delimiter=',')
        return


if __name__ == '__main__':
    # 导入数据
    aviris = import_data()
    X = aviris.X
    y = aviris.y

    # 划分训练集和测试集
    print("Spliting data...")
    X_train = X[:20000, :]
    y_train = y[:20000]
    X_test = X[20000:, :]
    y_test = y[20000:]

    # 利用 LDA 降维
    print("LDA transforming...")
    lda = LinearDiscriminantAnalysis(n_components=50)
    lda.fit(X_train, y_train)
    X_train = lda.transform(X_train)
    X_test = lda.transform(X_test)

    # 使用 SVM 分类
    print("Classifing...")
    svc = svm.SVC(kernel='rbf')
    svc.fit(X_train, y_train)
    train_score = svc.score(X_train, y_train)
    test_score = svc.score(X_test, y_test)
    print("Train score: ", train_score)
    print("Test score: ", test_score)

    # 保存模型
    print("Saving model...")
    joblib.dump(svc, "./clf.model")
    print("Model saved at: ./clf.model")
