import numpy as np
import matplotlib.pyplot as plt


class import_data(object):
    def __init__(self):
        self.name = np.loadtxt("data.csv", dtype=np.str, usecols=(1,), delimiter=',')
        self.place = np.loadtxt("data.csv", dtype=np.int32, usecols=(4,), delimiter=',')
        self.__init_X__()
        self.__init_y__()
        pass

    def __encode__(self, X):
        if len(X) == 0:
            return
        X_encode = []
        for x in X:
            sum = 0
            for c in x:
                sum = sum * 10 + ord(c)
            X_encode.append(sum)
        X_encode = np.array(X_encode, dtype=np.int32)
        return X_encode

    def __init_X__(self):
        name = self.__encode__(self.name).reshape((-1, 1))
        place = self.place.reshape((-1, 1))
        self.X = np.concatenate((name, place), axis=1)

    def __init_y__(self):
        scores = np.loadtxt("data.csv", dtype=np.float32, usecols=(8,), delimiter=',')
        y = []
        for score in scores:
            if score > 90:
                y.append('b')
            else:
                if score > 80:
                    y.append('c')
                else:
                    if score > 70:
                        y.append('g')
                    else:
                        if score > 60:
                            y.append('k')
                        else:
                            y.append('m')
        self.y = np.array(y)


if __name__ == '__main__':
    data = import_data()
    X = data.X
    y = data.y
    plt.scatter(X[:, 0], X[:, 1], c=y, marker='.')
    plt.show()
