#!/usr/bin/env python3

import numpy as np
import pandas as pd
import re
import os
# from scipy import spacial
import pickle
import tensorflow as tf
from tensorflow.contrib import rnn
import datetime
from os import listdir
from os.path import isfile, join
from random import randint

EMBEDDING_DIMENSION = 200
MAX_BLURB_LENGTH = 35
BATCH_SIZE = 24
LSTM_UNITS = 24
NUM_CLASSES = 2
ITERATIONS = 2000
LEARNING_RATE = .001
NUM_HIDDEN = 2
NUM_STEPS = 35

hyp_str = "blrbLen_" + str(MAX_BLURB_LENGTH) + "nLST_" + str(LSTM_UNITS) + "lr_" + str(LEARNING_RATE) + "n_hidd_" + "_"


def _getTrainBatch(train_succ_ids, train_fail_ids):
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


def _getTestBatch(test_succ_ids, test_fail_ids):
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
    with open('blurb_data.pickle', 'rb') as f:
        train_succ, train_fail, test_succ, test_fail, weights = pickle.load(f)

    # BUILD THE ACTUAL NN

    tf.reset_default_graph()

    # Set up placeholders for input and labels
    with tf.name_scope("Labels") as scope:
        labels = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_CLASSES])
    with tf.name_scope("Input") as scope:
        input_data = tf.placeholder(tf.int32, [BATCH_SIZE, MAX_BLURB_LENGTH])

    # Get embedding vector
    with tf.name_scope("Embeds_Layer") as scope:
        embedding = tf.Variable(tf.zeros([BATCH_SIZE, MAX_BLURB_LENGTH, EMBEDDING_DIMENSION]), dtype=tf.float32,
                                name='embedding')
        embed = tf.nn.embedding_lookup(weights, input_data)  # maybe change 'embedding back to 'weights'

    # Set up LSTM cell then wrap cell in dropout layer to avoid overfitting
    with tf.name_scope("LSTM_Cell") as scope:
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(LSTM_UNITS)
        lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=0.75)
        stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell for _ in range(NUM_HIDDEN)], state_is_tuple=True)
        initial_state = stacked_lstm.zero_state(BATCH_SIZE, dtype=tf.float32)

    with tf.name_scope("RNN_Forward") as scope:
        value, state = tf.nn.dynamic_rnn(stacked_lstm, embed, dtype=tf.float32, time_major=False)

    with tf.name_scope("Fully_Connected") as scope:
        weight = tf.Variable(tf.truncated_normal([LSTM_UNITS, NUM_CLASSES]), name='weights')
        bias = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES]), name='bias')
        value = tf.transpose(value, [1, 0, 2], name='last_lstm')
        last = tf.gather(value, int(value.get_shape()[0]) - 1)
        tf.summary.histogram("weights", weight)
        tf.summary.histogram("biases", bias)

    with tf.name_scope("Predictions") as scope:
        prediction = (tf.matmul(last, weight) + bias)

    # Cross entropy loss with a softmax layer on top
    # Using Adam for optimizer
    with tf.name_scope("Loss_and_Accuracy") as scope:
        correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
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
    # loss_summary = tf.summary.scalar("Loss", loss)
    acc_summary = tf.summary.scalar('Accuracy', accuracy)

    # Training summaries
    # train_summary_op = tf.summary.merge([loss_summary, acc_summary])
    train_summary_dir = os.path.join(logdir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    # Testing summaries
    # test_summary_op = tf.summary.merge([loss_summary, acc_summary])
    test_summary_dir = os.path.join(logdir, "summaries", "test")
    test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)

    summary_op = tf.summary.merge_all()
    # summary_op = tf.summary.merge([train_summary_op, test_summary_op])
    # Checkpointing
    checkpoint_dir = os.path.abspath(os.path.join(logdir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    # Tensorflow assumes this directory already exists so we need to create it
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    for i in range(ITERATIONS):
        # Next Batch of blurbs
        nextTrainBatch, nextTrainBatchLabels = _getTrainBatch(train_succ, train_fail);
        sess.run(optimizer, {input_data: nextTrainBatch, labels: nextTrainBatchLabels})

        # Write training summary to board
        if i % 50 == 0:
            summary = sess.run(summary_op, {input_data: nextTrainBatch, labels: nextTrainBatchLabels})
            train_summary_writer.add_summary(summary, i)
            train_summary_writer.flush()

            nextTestBatch, nextTestBatchLabels = _getTestBatch(test_succ, test_fail);
            testSummary = sess.run(summary_op, {input_data: nextTestBatch, labels: nextTestBatchLabels})
            test_summary_writer.add_summary(testSummary, i)
            test_summary_writer.flush()

        # Save network every so often
        if i % 1000 == 0 and i != 0:
            save_path = saver.save(sess, f"temp/models/_pretrained_lstm.ckpt", global_step=i)
            print(f"saved to {save_path}")

    train_summary_writer.close()
    test_summary_writer.close()


if __name__ == '__main__':
    main()
