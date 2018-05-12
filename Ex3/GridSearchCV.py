from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
import numpy as np

# Load database and normalize
database = np.loadtxt('Yale.csv', delimiter=',', dtype=np.float32)
X = database[:, :-1] / 255
y = database[:, -1]

# PCA
if X.shape[1] > 64:
    pca = PCA(n_components=64)
    X = pca.fit_transform(X)

para_set = {'kernel': ['rbf', 'poly', 'linear', 'sigmoid'],
            'C': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            'gamma': np.linspace(0, 0.01, 50)}
svc = svm.SVC()
clf = GridSearchCV(svc, para_set, cv=10)
clf.fit(X, y)
means = clf.cv_results_['mean_test_score']
for mean, params in zip(means, clf.cv_results_['params']):
    print("| %0.3f | %0.1f | %0.3f | %s" % (mean, params['C'], params['gamma'], params['kernel']))
print('========================================================')
joblib.dump(clf.best_estimator_, 'svm.model')
print('best score:', clf.best_score_)
print(clf.best_params_)
print('Model saved as svm.model')