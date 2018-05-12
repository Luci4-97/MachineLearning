import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class MLP(object):
    def __init__(self):
        # 网络参数
        self.learning_rate = 0.001  # 学习率
        self.n_hidden_1 = 1  # 第一层神经元个数
        self.n_input = 2  # 样本特征数
        # 定义权值和偏置
        self.Weights = {
            'h1': tf.Variable(tf.random_normal([self.n_input, self.n_hidden_1]), name='layer1_w'),
            'out': tf.ones([self.n_hidden_1, 1], dtype=tf.float32)
        }
        self.biases = {
            'h1': tf.Variable(tf.random_normal([1, self.n_hidden_1]), name='layer1_bias'),
            'out': tf.constant([0.])
        }
        self.model_path = "./model/model.ckpt"  # 模型保存路径
        self.names = ['h1', 'out']  # 便与遍历
        return

    def __add_layer__(self, name, inputs, activation_function=None):
        """
        添加一个神经网络层
        :param inputs: 输入数据
        :param activation_function: 激活函数
        :return: 该层输出
        """
        ys = tf.matmul(inputs, self.Weights[name]) + self.biases[name]
        if activation_function is None:
            outputs = ys
        else:
            outputs = activation_function(ys)
        return outputs

    def fit(self, X_train, y_train, max_iter=10000):
        """
        训练分类器
        :param X_train:训练样本
        :param y_train:训练标签
        :return:
        """
        X = tf.placeholder(tf.float32, [None, 2])
        y = tf.placeholder(tf.float32, [None, 1])
        # 第一层 使用sigmoid代替hardlimit
        layer1 = self.__add_layer__('h1', X, activation_function=None)
        # 输出层
        predict = self.__add_layer__('out', layer1, tf.nn.sigmoid)
        # 定义损失函数
        loss = tf.reduce_sum(tf.pow(predict - y, 2))
        # 定义优化器
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(loss)
        # 定义保存器
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            # 开始训练
            for epoch in range(max_iter):
                X_train = X_train.reshape((-1, 2))
                y_train = y_train.reshape((-1, 1))
                sess.run(optimizer, feed_dict={X: X_train, y: y_train})

                if (epoch + 1) % 20 == 0:
                    l = sess.run(loss, feed_dict={X: X_train, y: y_train})
                    print("Epoch:", '%04d' % (epoch + 1), "loss=", "{:.3f}".format(l))
            print("Optimization Finished!")
            training_loss = sess.run(loss, feed_dict={X: X_train, y: y_train})
            print("Training loss=", training_loss, '\n')
            res = np.around(np.abs(sess.run(predict, feed_dict={X: X_train}))).reshape((1, -1))
            print("Training result: ", res)
            saver.save(sess, self.model_path)
            print("Model saved at: ", self.model_path)
            print("""
            Input
                |-Weight:{0}
                |-bias:{1}
            layer1
                |-Weight:{2}
                |-bias:{3}
            output
            """.format(self.Weights['h1'].eval().reshape((1, -1)),
                       self.biases['h1'].eval(),
                       self.Weights['out'].eval().reshape((1, -1)),
                       self.biases['out'].eval()))
        return

    def get_params(self):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, self.model_path)  # 恢复模型
            weight = self.Weights.copy()
            bias = self.biases.copy()
            for name in self.names:
                weight[name] = self.Weights[name].eval()
                bias[name] = self.biases[name].eval()
        return weight, bias

    def predict(self, X_test):
        """
        使用模型预测
        :param X_test: 测试数据
        :return:
        """
        # 重建网络
        X = tf.placeholder(tf.float32, [None, 2])
        layer1 = self.__add_layer__('h1', X, activation_function=None)
        # 输出层
        predict = self.__add_layer__('out', layer1, tf.nn.sigmoid)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, self.model_path)  # 恢复模型
            print("""
            Input
                |-Weight:{0}
                |-bias:{1}
            layer1
                |-Weight:{2}
                |-bias:{3}
            output
            """.format(self.Weights['h1'].eval().reshape((1, -1)),
                       self.biases['h1'].eval(),
                       self.Weights['out'].eval().reshape((1, -1)),
                       self.biases['out'].eval()))
            result = np.around(np.abs(sess.run(predict, feed_dict={X: X_test}))).reshape((1, -1))[0]  # 预测
            print("[Result]: ", result)
        return result


def draw_region(X, y, weight, bias):
    """
    画出分类区域
    :param X: 测试数据，每个样本两个特征
    :return:
    """
    # 画图
    color = y.astype(np.str)
    color[np.where(y == 0)[0]] = 'r'
    color[np.where(y == 1)[0]] = 'c'
    plt.scatter(X[:, 0], X[:, 1], c=color, marker='.')
    weight = weight['h1']
    print(weight.reshape((1, -1)))
    print(bias['h1'])
    k = -(weight[0] / weight[1])
    b = -(bias['h1'] / weight[1])[0]
    x = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
    y = k * x + b
    plt.plot(x, y)
    plt.show()
    return


if __name__ == '__main__':
    X_train = np.loadtxt('train_data.csv', dtype=np.float, delimiter=',')
    y_train = np.loadtxt('train_label.csv', dtype=np.int, delimiter=',')
    X_test = np.loadtxt('test_data.csv', dtype=np.float, delimiter=',')
    y_test = np.loadtxt('test_label.csv', dtype=np.int, delimiter=',')
    mlp = MLP()
    # mlp.fit(X_train, y_train)
    result = mlp.predict(X_test)
    weight, bias = mlp.get_params()
    draw_region(X_test, result, weight, bias)
    # pause = 1
