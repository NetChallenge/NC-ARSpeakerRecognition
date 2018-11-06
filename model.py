import numpy
import librosa
import tensorflow as tf

class Model:

    def __init__(self, sess, name, output_size, learning_rate=0.001, keep_prob=0.7):
        self.sess = sess
        self.name = name
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self._build_net()


    def _build_net(self):
        with tf.variable_scope(self.name):

            self.keep_prob = tf.placeholder(tf.float32)

            self.X = tf.placeholder(tf.float32, [None, 40])

            self.Y = tf.placeholder(tf.float32, [None, self.output_size])


            W1 = tf.get_variable("W1", shape=[40, 500], initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.Variable(tf.random_normal([500]))
            L1 = tf.nn.sigmoid(tf.matmul(self.X , W1) + b1)
            L1 = tf.nn.dropout(L1, keep_prob=self.keep_prob)

            W2 = tf.get_variable("W2", shape=[500, 300], initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.Variable(tf.random_normal([300]))
            L2 = tf.nn.sigmoid(tf.matmul(L1, W2) + b2)
            L2 = tf.nn.dropout(L2, keep_prob=self.keep_prob)

            W3 = tf.get_variable("W3", shape=[300, 100], initializer=tf.contrib.layers.xavier_initializer())
            b3 = tf.Variable(tf.random_normal([100]))
            L3 = tf.nn.sigmoid(tf.matmul(L2, W3) + b3)
            L3 = tf.nn.dropout(L3, keep_prob=self.keep_prob)


            W4 = tf.get_variable("W4", shape=[100, 25], initializer=tf.contrib.layers.xavier_initializer())
            b4 = tf.Variable(tf.random_normal([25]))
            L4 = tf.nn.sigmoid(tf.matmul(L3, W4) + b4)
            L4 = tf.nn.dropout(L4, keep_prob=self.keep_prob)

            W5 = tf.get_variable("W5", shape=[25, self.output_size], initializer=tf.contrib.layers.xavier_initializer())
            b5 = tf.Variable(tf.random_normal([self.output_size]))
            self.hypothesis = tf.nn.relu(tf.matmul(L4, W5) + b5)

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.hypothesis, labels=self.Y))

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        correct_prediction = tf.equal(tf.argmax(self.hypothesis,1), tf.argmax(self.Y, 1))

        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test, keep_prob=1.0):
        return  self.sess.run(self.hypothesis, feed_dict={
            self.X: x_test, self.keep_prob: keep_prob})

    def get_accuracy(self, x_test, y_test, keep_prob = 1.0):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.keep_prob: keep_prob})

    def train(self, x_data, y_data, keep_prob = 0.7):
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.X: x_data, self.Y: y_data, self.keep_prob: keep_prob})













