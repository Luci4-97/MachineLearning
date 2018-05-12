import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def import_data():
    """
    从./data.csv 导入数据
    :return: [x,y]
    """
    data = np.loadtxt('data.csv', dtype=np.float32, delimiter=',')
    x = data[:, 0]
    y = data[:, 1]
    return x, y


def train(X, max_iter=4000, learn_rate=1e-2):
    """
    训练模型
    :param X: 数据
    :param max_iter: 最大迭代次数，默认6000
    :param learn_rate: 梯度下降步长，默认1e-2
    :return:
    """
    n_samples = X.shape[0]  # 样本数

    # 声明占位符
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)

    # 定义参数
    w = tf.Variable(np.random.randn(), name='weight', dtype=tf.float32)
    b = tf.Variable(np.random.randn(), name='bias', dtype=tf.float32)

    # 定义模型 y=wx+b
    predict = tf.add(tf.multiply(w, x), b)

    # 定义损失
    loss = tf.reduce_sum(tf.pow(predict - y, 2)) / (2 * n_samples)

    # 定义优化器 梯度下降
    optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss)

    # 初始化变量
    init = tf.initialize_all_variables()

    # 训练
    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(max_iter):
            for (train_x, train_y) in zip(X[:, 0], X[:, 1]):
                sess.run(optimizer, feed_dict={x: train_x, y: train_y})

            if (epoch + 1) % 20 == 0:
                l = sess.run(loss, feed_dict={x: X[:, 0], y: X[:, 1]})
                print("Epoch:", '%04d' % (epoch + 1), "loss=", "{:.3f}".format(l), "w=", sess.run(w), "b=", sess.run(b))

        print("Optimization Finished!")
        training_loss = sess.run(loss, feed_dict={x: X[:, 0], y: X[:, 0]})
        print("Training loss=", training_loss, "w=", sess.run(w), "b=", sess.run(b), '\n')
        weight = sess.run(w)
        bias = sess.run(b)
    return weight, bias


def draw(weight, bias, X):
    # 画图
    plt.plot(X[:, 0], X[:, 1], 'c.', label="Original data")
    plt.plot(X[:, 0], weight * X[:, 0] + bias, label="Fitted line")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    X = np.loadtxt('data.csv', dtype=np.float32, delimiter=',')
    weight, bias = train(X)
    draw(weight, bias, X)
