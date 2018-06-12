"""
Deep Learning Assignment II
Task 5b
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.model_selection import train_test_split

tf.logging.set_verbosity(tf.logging.INFO)


def main(args):
    mnist = input_data.read_data_sets("data/", reshape=False, one_hot=True, validation_size=0)

    learning_rate = 0.001
    training_iterations = 5000
    batch_size = 100
    display_iterations = 1
    early_stop_counter = 3

    input_layer = tf.placeholder(tf.float32, [None, 28, 28, 1])
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

    with tf.name_scope('Adam'):
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        avg_cost = 0.
        avg_val_acc = 0.
        X_train, X_validation, y_train, y_validation = train_test_split(mnist.train.images, mnist.train.labels,
                                                                        test_size=0.2)
        no_improvement_counter = 0
        last_val_acc = 0
        iterations_per_epoch = float(len(X_train)) / batch_size
        for iteration in range(training_iterations):
            start_index = int((iteration % iterations_per_epoch) * batch_size)
            end_index = int(((iteration % iterations_per_epoch) + 1) * batch_size)
            batch_xs, batch_ys = X_train[start_index:end_index], y_train[start_index: end_index]
            _, c = sess.run([optimizer, cost],
                            feed_dict={input_layer: batch_xs, output_layer: batch_ys, training: True})
            avg_cost += c / display_iterations
            val_acc = accuracy.eval({input_layer: X_validation, output_layer: y_validation, training: False})
            avg_val_acc += val_acc / display_iterations
            if (iteration + 1) % display_iterations == 0:
                tf.logging.info("Iteration: %s\tcost=%s\tvalidation accuracy=%s", (iteration + 1), avg_cost,
                                avg_val_acc)
                avg_cost = 0.
                avg_val_acc = 0.
            if last_val_acc >= val_acc:
                no_improvement_counter += 1
            else:
                no_improvement_counter = 0
            last_val_acc = val_acc
            if no_improvement_counter == early_stop_counter:
                tf.logging.info(
                    "Iteration %s Early stop, There was no accuracy improvement in the last 3 batches" % (
                        iteration + 1))
                break

        tf.logging.info("Validation Accuracy: %s",
                        accuracy.eval({input_layer: X_validation, output_layer: y_validation, training: False}))
        tf.logging.info("Accuracy: %s",
                        accuracy.eval(
                            {input_layer: mnist.test.images, output_layer: mnist.test.labels, training: False}))


if __name__ == "__main__":
    tf.app.run()
