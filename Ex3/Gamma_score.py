from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt

# Load database and normalize
database = np.loadtxt('Yale.csv', delimiter=',', dtype=np.float32)
X = database[:, :-1] / 255
y = database[:, -1]

# PCA
pca = PCA(n_components=64)
X = pca.fit_transform(X)

# optimize parameters(grid search) and plot Gamma_Score image
rbf_x = []
rbf_y = []
poly_x = []
poly_y = []
sigmoid_x = []
sigmoid_y = []
linear_x = []
linear_y = []
para_set = {'kernel': ['rbf', 'poly', 'sigmoid', 'linear'],
            'gamma': np.linspace(0, 0.1, 100)}
svc = svm.SVC(C=2)
gs = GridSearchCV(svc, para_set, cv=10)  # cv: cross validation. 10-fold (default = 3)
gs.fit(X, y)
mean_score = gs.cv_results_['mean_test_score']
params = gs.cv_results_['params']
for score, param in zip(mean_score, params):
    print("| %0.3f | %0.4f | %s" % (score, param['gamma'], param['kernel']))
    if param['kernel'] == 'rbf':
        rbf_x.append(param['gamma'])
        rbf_y.append(score)
    if param['kernel'] == 'poly':
        poly_x.append(param['gamma'])
        poly_y.append(score)
    if param['kernel'] == 'sigmoid':
        sigmoid_x.append(param['gamma'])
        sigmoid_y.append(score)
    if param['kernel'] == 'linear':
        linear_x.append(param['gamma'])
        linear_y.append(score)
print('========================================================')
print('best score:', gs.best_score_)
rbf_x = np.array(rbf_x, dtype=np.float32)
rbf_y = np.array(rbf_y, dtype=np.float32)
poly_x = np.array(poly_x, dtype=np.float32)
poly_y = np.array(poly_y, dtype=np.float32)
sigmoid_x = np.array(sigmoid_x, dtype=np.float32)
sigmoid_y = np.array(sigmoid_y, dtype=np.float32)
linear_x = np.array(sigmoid_x, dtype=np.float32)
linear_y = np.array(sigmoid_y, dtype=np.float32)
plt.title('Gamma-score Relationship')
plt.axis([0, 0.1, 0.0, 1.0])
plt.plot(rbf_x, rbf_y, color='green', label='rbf')
plt.plot(poly_x,poly_y, color='red', label='poly')
plt.plot(sigmoid_x, sigmoid_y, color='skyblue', label='sigmoid')
plt.plot(linear_x, linear_y, color='yellow', label='linear')
plt.xlabel('Gamma')
plt.ylabel('score')
plt.legend()
plt.savefig('Gamma-score.png')
plt.show()




