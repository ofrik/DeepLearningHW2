# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 09:37:48 2018

@author: Lenovo
"""

"""
Deep Learning Assignment II
Task 3
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.logging.set_verbosity(tf.logging.INFO)


def main(args):
    mnist = input_data.read_data_sets("data/", reshape=False, one_hot=True, validation_size=0)

    learning_rate = 0.001
    training_iterations = 5000
    batch_size = 100
    display_iterations = 250

    input_layer = tf.placeholder(tf.float32, [None, 28, 28, 1])
    # input_layer = tf.reshape(raw_data, [-1, 28, 28, 1])
    output_layer = tf.placeholder(tf.float32, [None, 10])
    training = tf.placeholder(tf.bool)

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    norm1 = tf.nn.local_response_normalization(conv1)

    pool1 = tf.layers.max_pooling2d(inputs=norm1, pool_size=[2, 2], strides=2)

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    norm2 = tf.nn.local_response_normalization(conv2)

    pool2 = tf.layers.max_pooling2d(inputs=norm2, pool_size=[2, 2], strides=2)

    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    dense1 = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense1, rate=0.4, training=training)

    logits = tf.layers.dense(inputs=dropout, units=10, activation=tf.nn.relu)

    with tf.name_scope('Model'):
        predictions = tf.nn.softmax(logits)

    with tf.name_scope('Loss'):
        cost = tf.reduce_mean(-tf.reduce_sum(output_layer * tf.log(predictions), reduction_indices=1))

    with tf.name_scope('Accuracy'):
        accuracy = tf.equal(tf.argmax(predictions, 1), tf.argmax(output_layer, 1))
        accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))

    with tf.name_scope('SGD'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        avg_cost = 0.
        for iteration in range(training_iterations):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, cost],
                            feed_dict={input_layer: batch_xs, output_layer: batch_ys, training: True})
            avg_cost += c / display_iterations
            if (iteration + 1) % display_iterations == 0:
                tf.logging.info("Iteration: %s\tcost=%s", (iteration + 1), avg_cost)
                avg_cost = 0.

        tf.logging.info("Accuracy: %s",
                        accuracy.eval(
                            {input_layer: mnist.test.images, output_layer: mnist.test.labels, training: False}))


if __name__ == "__main__":
    tf.app.run()
