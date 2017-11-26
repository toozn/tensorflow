import numpy as np
import tensorflow as tf
import input_data
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
def conv_2d(x, W):
    return tf.nn.conv2d(x, W, [1, 1, 1, 1], padding='SAME')

def max_pooling_2d(X):
    return tf.nn.max_pool(X, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

class dataset:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.train_image = x_train.tolist()
        self.train_label = y_train.tolist()
        self.test_image = x_test.tolist()
        self.test_label = y_test.tolist()
        self.train_size = len(self.train_label)
        self.test_size = len(self.test_label)
        mmax = max(y_train)[0]
        for item in self.train_label:
            tp = item[0]
            item = [0 for x in range(mmax + 1)]
            item[tp] = 1
        for item in self.test_label:
            tp = item[0]
            item = [0 for x in range(mmax + 1)]
            item[tp] = 1

    def batch(self, batch_size, mode):
        res = [[], []]
        if mode == 'train':
            st = np.random.randint(0, self.train_size - batch_size)
            for i in range(st, st + batch_size):
                res[0].append(self.train_image[i])
                res[1].append(self.train_label[i])
        elif mode == 'test':
            st = np.random.randint(0, self.test_size - batch_size)
            for i in range(st, st + batch_size):
                res[0].append(self.test_image[i])
                res[1].append(self.test_label[i])
        print res[1]
        return res

def main():
    batch_size = 128
    epoch_size = 20
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    ds = dataset(x_train, y_train, x_test, y_test)
    sess = tf.Session()
    X = tf.placeholder(tf.float32, [None, 32, 32, 3])
    Y = tf.placeholder(tf.float32, [None, 10])
    #image = tf.reshape(X, [-1, 32, 32, 3])

    W1 = tf.get_variable('W1', [5, 5, 3, 32], tf.float32, tf.contrib.layers.xavier_initializer_conv2d())
    b1 = tf.get_variable('b1', [32], tf.float32, tf.constant_initializer(0.01))
    c1 = tf.nn.relu(conv_2d(X, W1) + b1)
    p1 = max_pooling_2d(c1)

    W2 = tf.get_variable('W2', [5, 5, 32, 64], tf.float32, tf.contrib.layers.xavier_initializer_conv2d())
    b2 = tf.get_variable('b2', [64], tf.float32, tf.constant_initializer(0.01))
    c2 = tf.nn.relu(conv_2d(p1, W2) + b2)
    p2 = max_pooling_2d(c2)

    fc_in = tf.reshape(p2, [-1, 7 * 7 * 64])
    W3 = tf.get_variable('W3', [7 * 7 * 64, 1024], tf.float32, tf.contrib.layers.xavier_initializer_conv2d())
    b3 = tf.get_variable('b3', [1024], tf.float32, tf.constant_initializer(0.01))
    fc1 = tf.nn.relu(tf.matmul(fc_in, W3) + b3)

    keep_prob = tf.placeholder(tf.float32)
    dp = tf.nn.dropout(fc1, keep_prob)
    
    W4 = tf.get_variable('W4', [1024, 10], tf.float32, tf.contrib.layers.xavier_initializer_conv2d())
    b4 = tf.get_variable('b4', [10], tf.float32, tf.constant_initializer(0.01))
    final = tf.nn.softmax(tf.matmul(dp, W4) + b4)
    pre = tf.equal(tf.arg_max(Y, 1), tf.arg_max(final, 1))
    acc = tf.reduce_mean(tf.cast(pre, tf.float32))
    cross_entropy = -tf.reduce_sum(Y * tf.log(final))
    optimizer = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
    sess.run(tf.global_variables_initializer())

    for epoch in range(epoch_size):
        for step in range(100):
            batch = ds.batch(batch_size, 'train')
            _, loss = sess.run([optimizer, cross_entropy], feed_dict={X: batch[0], Y: batch[1], keep_prob: 0.5})
            if step % 100 == 99:
                print "epoch %d step %d: loss %.10f" % (epoch, step, loss)
        batch = ds.batch(1000, 'test')
        acc_test = sess.run(acc, feed_dict={X: batch[0], Y: batch[1], keep_prob: 1.0})
        print "epoch %d test acc %.10f" % (epoch, acc_test)

main()