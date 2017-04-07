import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


class ModelConfig:
    hidden_size = 256
    output_size = 10
    initial_lr = 1e-3
    epochs = 500
    momentum = 0.9
    decay = 0.9
    beta1 = 0.9
    beta2 = 0.99
    batch_size = 50


def build_graph(config, optimizer, param):
    data_x = tf.placeholder(tf.float32, [None, 784])
    data_y = tf.placeholder(tf.float32, [None, 10])
    hidden_weights = tf.Variable(
                                tf.truncated_normal([784, config.hidden_size],
                                                    stddev=0.1)
                            )
    hidden_bias = tf.Variable(
        tf.truncated_normal([config.hidden_size], stddev=0.1)
    )
    hidden_layer = tf.matmul(data_x, hidden_weights) + hidden_bias
    output_weights = tf.Variable(
                                tf.truncated_normal([config.hidden_size, 10],
                                                    stddev=0.1)
                            )
    output_bias = tf.Variable(
        tf.truncated_normal([10], stddev=0.1)
    )
    output_layer = tf.matmul(hidden_layer, output_weights) + output_bias
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=data_y,
                                                logits=output_layer))
    optimize = optimizer(**param).minimize(loss)
    correct_prediction = tf.equal(
                            tf.argmax(output_layer, 1),
                            tf.argmax(data_y, 1)
                        )
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return data_x, data_y, loss, optimize, accuracy


def build_train(config, optimizer, param, mnist, label, color):
    x, y, loss, optimize, accuracy = build_graph(config, optimizer, param)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    train_acc = []
    test_acc = []
    for i in range(config.epochs):
        batch = mnist.train.next_batch(config.batch_size)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(
                feed_dict={x: mnist.train.images, y: mnist.train.labels}
            )
            test_accuracy = accuracy.eval(
                feed_dict={x: mnist.test.images, y: mnist.test.labels}
            )
            print("step %d, training accuracy %g, test accuracy %g"
                  % (i, train_accuracy, test_accuracy))
            train_acc.append(train_accuracy)
            test_acc.append(test_accuracy)
        optimize.run(feed_dict={x: batch[0], y: batch[1]})

    print("test accuracy %g" % accuracy.eval(
        feed_dict={x: mnist.test.images, y: mnist.test.labels}))
    length = len(train_acc)
    t = np.array(range(1, length + 1))*100
    plt.plot(t, train_acc, label=label+"_train", color=color, linestyle='-')
    plt.plot(t, test_acc, label=label+"_test", color=color, linestyle='--')


def main():
    mnist = input_data.read_data_sets("./MNIST", one_hot=True)
    config = ModelConfig()

    # SGD optimizer
    optimizer_sgd = tf.train.GradientDescentOptimizer
    param_sgd = {'learning_rate': config.initial_lr}
    build_train(config, optimizer_sgd, param_sgd, mnist,
                "SGD",
                "green")

    # Momentum optimizer
    optimizer_momentum = tf.train.MomentumOptimizer
    param_momentum = {'learning_rate': config.initial_lr,
                      'momentum': config.momentum}
    build_train(config, optimizer_momentum, param_momentum, mnist,
                "Momentum", "orange")

    # AdaGrad
    optimizer_adagrad = tf.train.AdagradOptimizer
    param_adagrad = {'learning_rate': config.initial_lr}
    build_train(config, optimizer_adagrad, param_adagrad, mnist,
                "AdaGrad",
                "yellow")

    # NesterovMomentum
    optimizer_nest_momen = tf.train.MomentumOptimizer
    param_nest_momen = {'learning_rate': config.initial_lr,
                        'momentum': config.momentum,
                        'use_nesterov': True}
    build_train(config, optimizer_nest_momen, param_nest_momen, mnist,
                "NesterovMomentum",
                "red")

    # RMSProp
    optimizer_rms = tf.train.RMSPropOptimizer
    param_rms = {'learning_rate': config.initial_lr,
                 'momentum': config.momentum,
                 'decay': config.decay}
    build_train(config, optimizer_rms, param_rms, mnist,
                "RMSProp",
                "blue")

    # ADAM
    optimizer_adam = tf.train.AdamOptimizer
    param_adam = {'learning_rate': config.initial_lr,
                  'beta1': config.beta1,
                  'beta2': config.beta2}
    build_train(config, optimizer_adam, param_adam, mnist,
                "ADAM",
                "black")
    plt.legend()
    plt.xlabel('iteration')
    plt.ylabel('accuracy')
    plt.show()


if __name__ == "__main__":
    main()
