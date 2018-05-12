from sklearn import datasets
from sklearn import svm
from sklearn.externals import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV

if __name__ == '__main__':
    # 从sklearn导入数据IRIS数据
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # 初始化SVM分类器
    svc = svm.SVC()

    # 设置参数网格，用于网格搜索法调参
    para_set = {'kernel': ['rbf', 'poly', 'linear', 'sigmoid'],
                'C': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
                'gamma': np.linspace(0, 0.01, 50)}

    # 初始化网格优化器
    gs = GridSearchCV(svc, para_set, cv=10)

    # 训练
    gs.fit(X, y)
    means = gs.cv_results_['mean_test_score']
    for mean, params in zip(means, gs.cv_results_['params']):
        print("| %0.3f | %0.1f | %0.3f | %s" % (mean, params['C'], params['gamma'], params['kernel']))
    print('========================================================')
    joblib.dump(gs.best_estimator_, 'svm.model')
    print('best score:', gs.best_score_)
    print(gs.best_params_)
    print('Model saved as svm.model')
