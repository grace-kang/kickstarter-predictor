#!/usr/bin/env python3

import numpy as np
import os
import pickle
import tensorflow as tf
import datetime
from random import randint

EMBEDDING_DIMENSION = 300
MAX_BLURB_LENGTH = 35
BATCH_SIZE = 24
LSTM_UNITS = 2
NUM_CLASSES = 2
ITERATIONS = 1000
LEARNING_RATE = 1e-2
NUM_HIDDEN = 1
NUM_STEPS = 35
DROPOUT_KEEP_PROB = .65

hyp_str = "nLST-" + str(LSTM_UNITS) + "__lr-" + str(LEARNING_RATE) + \
          "__n_hidd-" + str(NUM_HIDDEN) + \
          "__dOut-" + str(DROPOUT_KEEP_PROB * 100) + "_"


def _get_train_batch(train_succ_ids, train_fail_ids):
    labels = []
    arr = np.zeros([BATCH_SIZE, MAX_BLURB_LENGTH])
    for i in range(BATCH_SIZE):
        if i % 2 == 0:
            num = randint(1, len(train_succ_ids))
            arr[i] = train_succ_ids[num - 1:num]
            labels.append([1, 0])
        else:
            num = randint(1, len(train_fail_ids))
            arr[i] = train_fail_ids[num - 1:num]
            labels.append([0, 1])
    return arr, labels


def _get_test_batch(test_succ_ids, test_fail_ids):
    labels = []
    arr = np.zeros([BATCH_SIZE, MAX_BLURB_LENGTH])
    for i in range(BATCH_SIZE):
        if i % 2 == 0:
            num = randint(1, len(test_succ_ids))
            arr[i] = test_succ_ids[num - 1:num]
            labels.append([1, 0])
        else:
            num = randint(1, len(test_fail_ids))
            arr[i] = test_fail_ids[num - 1:num]
            labels.append([0, 1])
    return arr, labels


def main():
    # load up the saved ids and weights
    with open('pre_processed_files/blurb_data_9.pickle', 'rb') as f:
        train_succ, train_fail, test_succ, test_fail, weights = pickle.load(f)

    # BUILD THE ACTUAL NN

    tf.reset_default_graph()

    # Set up placeholders for input and labels
    with tf.name_scope("Labels") as scope:
        labels = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_CLASSES])
    with tf.name_scope("Input"):
        input_data = tf.placeholder(tf.int32, [BATCH_SIZE, MAX_BLURB_LENGTH])

    # Get embedding vector
    with tf.name_scope("Embeds_Layer") as scope:
        embedding = tf.Variable(tf.zeros([BATCH_SIZE, MAX_BLURB_LENGTH, EMBEDDING_DIMENSION]), dtype=tf.float32,
                                name='embedding')
        embed = tf.nn.embedding_lookup(weights, input_data)

    with tf.name_scope("LSTM_Cell") as scope:
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(LSTM_UNITS)
        lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=DROPOUT_KEEP_PROB)
        stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell for _ in range(NUM_HIDDEN)], state_is_tuple=True)
        initial_state = stacked_lstm.zero_state(BATCH_SIZE, dtype=tf.float32)

    with tf.name_scope("RNN_Forward") as scope:
        value, state = tf.nn.dynamic_rnn(stacked_lstm, embed,
                                         initial_state=initial_state,
                                         dtype=tf.float32, time_major=False)

    with tf.name_scope("Fully_Connected") as scope:
        weight = tf.Variable(tf.truncated_normal([LSTM_UNITS, NUM_CLASSES]), name='weights')
        bias = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES]), name='bias')
        value = tf.transpose(value, [1, 0, 2], name='last_lstm')
        last = tf.gather(value, int(value.get_shape()[0]) - 1)
        tf.summary.histogram("weights", weight)
        tf.summary.histogram("biases", bias)

    with tf.name_scope("Predictions")as scope:
        prediction = (tf.matmul(last, weight) + bias)

    # Cross entropy loss with a softmax layer on top
    # Using Adam for optimizer
    with tf.name_scope("Loss_and_Accuracy") as scope:
        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))

    with tf.name_scope("Training") as scope:
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    # Output directory for models and summaries
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    name = hyp_str + timestamp
    logdir = os.path.abspath(os.path.join(os.path.curdir, "temp/tboard", name))
    print(f"Writing to {logdir}")

    # Summaries for loss and accuracy
    acc_summary = tf.summary.scalar('Accuracy', accuracy)

    # Training summaries
    train_summary_dir = os.path.join(logdir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    # Testing summaries
    test_summary_dir = os.path.join(logdir, "summaries", "test")
    test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)

    summary_op = tf.summary.merge_all()

    checkpoint_dir = os.path.abspath(os.path.join(logdir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")

    # Tensorflow assumes this directory already exists so we need to create it
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    for i in range(ITERATIONS):
        # Next Batch of blurbs
        next_train_batch, next_train_batch_labels = _get_train_batch(train_succ, train_fail);
        sess.run(optimizer, {input_data: next_train_batch, labels: next_train_batch_labels})

        # Write training summary to board
        if i % 100 == 0:
            summary = sess.run(summary_op, {input_data: next_train_batch, labels: next_train_batch_labels})
            train_summary_writer.add_summary(summary, i)
            train_summary_writer.flush()

            next_test_batch, next_test_batch_labels = _get_test_batch(test_succ, test_fail);
            testSummary = sess.run(summary_op, {input_data: next_test_batch, labels: next_test_batch_labels})
            test_summary_writer.add_summary(testSummary, i)
            test_summary_writer.flush()

        # Save network every so often
        if i % 100 == 0 and i != 0:
            save_path = saver.save(sess, f"temp/models/_pretrained_lstm.ckpt", global_step=i)
            print(f"saved to {save_path}")

    train_summary_writer.close()
    test_summary_writer.close()


if __name__ == '__main__':
    main()
