import tensorflow as tf
import numpy as np


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


data_x = np.array([[8, 8, 0, 9],
                   [7, 1, 1, 1],
                   [2, 1, 7, 2],
                   [6, 6, 6, 6],
                   [1, 1, 1, 1],
                   [3, 2, 1, 3],
                   [7, 6, 6, 2],
                   [9, 3, 1, 3],
                   [0, 0, 0, 0],
                   [2, 2, 2, 2],
                   [3, 3, 3, 3],
                   [3, 5, 5, 5],
                   [8, 9, 1, 3],
                   [8, 0, 9, 6],
                   [7, 7, 7, 7],
                   [9, 9, 9, 9],
                   [7, 7, 5, 6],
                   [6, 8, 5, 5],
                   [9, 8, 8, 1],
                   [5, 5, 3, 1]])

data_y = np.reshape(np.array([6, 0, 0, 4, 0, 0, 2, 1, 4, 0,
                              0, 0, 3, 5, 0, 4, 1, 3, 5, 0]), [20, 1])

x = tf.placeholder(tf.int32, [None, 4])
y = tf.placeholder(tf.int32, [None, 1])
sess = tf.InteractiveSession()

x_onehot = tf.reshape(tf.one_hot(x, depth=10, axis=-1), [-1, 40])
y_onehot = tf.reshape(tf.one_hot(y, depth=10, axis=-1), [-1, 10])
weight1 = weight_variable([40, 100])
bias1 = bias_variable([100])

weight2 = weight_variable([100, 100])
bias2 = bias_variable([100])

weight3 = weight_variable([100, 10])
bias3 = bias_variable([10])

h1 = tf.nn.relu(tf.matmul(x_onehot, weight1) + bias1)

h2 = tf.nn.relu(tf.matmul(h1, weight2) + bias2)
output_logit = tf.matmul(h2, weight3) + bias3
output = tf.nn.softmax(output_logit)
loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(
        labels=y_onehot,
        logits=output_logit
        )
    )
# loss = tf.reduce_sum(tf.abs(tf.subtract(y_onehot, output)))

train_step = tf.train.AdamOptimizer(1e-6).minimize(loss)
sess.run(tf.global_variables_initializer())

for i in range(50000):
    train_loss, _ = sess.run(
        [loss, train_step],
        feed_dict={x: data_x, y: data_y}
    )
    print("step %d, loss %g" % (i, train_loss))

res = sess.run(output, feed_dict={x: data_x, y: data_y})
print(res)
print(np.argmax(np.reshape(res, [20, -1]), axis=1))
