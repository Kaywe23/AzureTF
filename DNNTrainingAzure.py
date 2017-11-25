import tensorflow as tf
import pickle
import numpy as np
from tensorflow.python.lib.io import file_io
import argparse
from nltk.tokenize import word_tokenize
import nltk
import math as math
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import io
import csv
import pandas as pd
import sys

reload(sys)
sys.setdefaultencoding('latin-1')

lemmatizer = WordNetLemmatizer()

hidden_nodes_1 = 2638
hidden_nodes_2 = int(hidden_nodes_1 * 0.5)
hidden_nodes_3 = int(hidden_nodes_1 * np.power(0.5, 2))
hidden_nodes_4 = int(hidden_nodes_1 * np.power(0.5, 3))
hidden_nodes_5 = int(hidden_nodes_1 * np.power(0.5, 4))
beta=0.001
n_classes = 2
batch_size = 100
hm_epochs =2
batch_count = 500
display_step = 1


graph = tf.Graph()
with graph.as_default():

    tf_train_dataset = tf.placeholder('float')
    tf_train_labels = tf.placeholder('float')

    # Hidden RELU layer 1
    weights_1 = tf.Variable(tf.truncated_normal([2638, hidden_nodes_1], stddev=math.sqrt(2.0 / (2638))))
    biases_1 = tf.Variable(tf.random_normal([hidden_nodes_1]))

    # Hidden RELU layer 2
    weights_2 = tf.Variable(tf.truncated_normal([hidden_nodes_1, hidden_nodes_2], stddev=math.sqrt(2.0 / hidden_nodes_1)))
    biases_2 = tf.Variable(tf.random_normal([hidden_nodes_2]))

    # Hidden RELU layer 3
    weights_3 = tf.Variable(tf.truncated_normal([hidden_nodes_2, hidden_nodes_3], stddev=math.sqrt(2.0 / hidden_nodes_2)))
    biases_3 = tf.Variable(tf.random_normal([hidden_nodes_3]))

    # Hidden RELU layer 4
    weights_4 = tf.Variable(tf.truncated_normal([hidden_nodes_3, hidden_nodes_4], stddev=math.sqrt(2.0 / hidden_nodes_3)))
    biases_4 = tf.Variable(tf.random_normal([hidden_nodes_4]))

    # Hidden RELU layer 5
    weights_5 = tf.Variable(tf.truncated_normal([hidden_nodes_4, hidden_nodes_5], stddev=math.sqrt(2.0 / hidden_nodes_4)))
    biases_5 = tf.Variable(tf.random_normal([hidden_nodes_5]))

    # Output layer
    weights_6 = tf.Variable(tf.truncated_normal([hidden_nodes_5, n_classes], stddev=math.sqrt(2.0 / hidden_nodes_5)))
    biases_6 = tf.Variable(tf.random_normal([n_classes]))


    # Hidden RELU layer 1
    logits_1 = tf.matmul(tf_train_dataset, weights_1) + biases_1
    hidden_layer_1 = tf.nn.relu(logits_1)

    # Dropout on hidden layer: RELU layer
    keep_prob = tf.placeholder("float")
    hidden_layer_1_dropout = tf.nn.dropout(hidden_layer_1, keep_prob)

    # Hidden RELU layer 2
    logits_2 = tf.matmul(hidden_layer_1_dropout, weights_2) + biases_2
    hidden_layer_2 = tf.nn.relu(logits_2)

    # Dropout on hidden layer: RELU layer
    hidden_layer_2_dropout = tf.nn.dropout(hidden_layer_2, keep_prob)

    # Hidden RELU layer 3
    logits_3 = tf.matmul(hidden_layer_2_dropout, weights_3) + biases_3
    hidden_layer_3 = tf.nn.relu(logits_3)

    # Dropout on hidden layer: RELU layer
    hidden_layer_3_dropout = tf.nn.dropout(hidden_layer_3, keep_prob)

    # Hidden RELU layer 4
    logits_4 = tf.matmul(hidden_layer_3_dropout, weights_4) + biases_4
    hidden_layer_4 = tf.nn.relu(logits_4)

    # Dropout on hidden layer: RELU layer
    hidden_layer_4_dropout = tf.nn.dropout(hidden_layer_4, keep_prob)

    # Hidden RELU layer 5
    logits_5 = tf.matmul(hidden_layer_4_dropout, weights_5) + biases_5
    hidden_layer_5 = tf.nn.relu(logits_5)

    # Dropout on hidden layer: RELU layer
    hidden_layer_5_dropout = tf.nn.dropout(hidden_layer_5, keep_prob)

    # Output layer
    logits_6 = tf.matmul(hidden_layer_5_dropout, weights_6) + biases_6


    # Minimize error using cross entropy
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_6, labels=tf_train_labels))

    # Loss function with L2 Regularization with decaying learning rate beta=0.5
    regularizers = tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(weights_2) + \
                   tf.nn.l2_loss(weights_3) + tf.nn.l2_loss(weights_4) + \
                   tf.nn.l2_loss(weights_5) + tf.nn.l2_loss(weights_6)
    cost = tf.reduce_mean(cost + beta * regularizers)

    # Gradient Descent
    # optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
    # Decaying learning rate
    global_step = tf.Variable(0)  # count the number of steps taken.
    start_learning_rate = 0.001
    learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, 100000, 0.96, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, global_step=global_step)

    # Predictions for the training
    train_prediction = tf.nn.softmax(logits_6)

    # define Checkpoint
    saver = tf.train.Saver()


    # summary
    correct = tf.equal(tf.argmax(logits_6, 1), tf.argmax(tf_train_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    cost_summary = tf.summary.scalar("cost", cost)
    acc_summary = tf.summary.scalar("accuracy", accuracy)

    summary_op = tf.summary.merge_all()




def trainDNN(train_file='lexikon2.pickle', csv_file='train_converted_vermischt.csv',
             csv_file2='vector_test_converted.csv', job_dir='./DNNTrainingAzure',
             checkpoint='model.ckpt', logs='tf.log', **args):
    file_stream = file_io.FileIO(train_file, mode='r')
    lexikon = pickle.load(file_stream)

    # valid dataset
    '''feature_sets = []
    labels = []
    with tf.gfile.Open(csv_file2, 'rb') as gc_file:
        lines2 = gc_file.readlines()

        for zeile in lines2:
            try:
                features = list(eval(zeile.split('::')[0]))
                label = list(eval(zeile.split('::')[1]))
                feature_sets.append(features)
                labels.append(label)
            except:
                pass

        test_1 = np.array(feature_sets)
        valid_dataset = test_1

        test_2 = np.array(labels)
        valid_labels = test_2'''



    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        summary_writer = tf.summary.FileWriter(job_dir, graph=graph)
        print('Start Training')
        try:
            epoch = int(tf.gfile.Open(logs, 'r').read().split('\n')[-2]) + 1
            print('START:', epoch)
        except:
            epoch = 1

        for epoch in range(hm_epochs):
            #if epoch != 1:
                #saver.restore(sess, checkpoint)
            avg_cost = 0.

            with tf.gfile.Open(csv_file, 'rb') as gcs_file:
                lines = gcs_file.readlines()

                zaehler = 0
                for zeile in lines:
                    zaehler += 1
                    label = zeile.split(':::')[0]
                    tweet = zeile.split(':::')[1]
                    woerter = word_tokenize(tweet.lower())
                    woerter = [lemmatizer.lemmatize(i) for i in woerter]
                    features = np.zeros(len(lexikon))
                    for wort in woerter:
                        if wort.lower() in lexikon:
                            indexWert = lexikon.index(wort.lower())
                            features[indexWert] += 1

                    train_dataset = np.array([list(features)])
                    train_labels = np.array([eval(label)])

                    feed_dict = {tf_train_dataset: np.array(train_dataset), tf_train_labels: np.array(train_labels),
                                 keep_prob: 0.5}
                    _, c, predictions, summary = sess.run([optimizer, cost, train_prediction, summary_op],
                                                          feed_dict=feed_dict)
                    writer.add_summary(summary, epoch * batch_count + zaehler)

                    avg_cost += c / batch_count

                    if zaehler >= batch_count:
                        break

                    if (zaehler % 100 == 0):
                        print("Minibatch loss at step {}: {}".format(zaehler, c))
                        #print("Minibatch accuracy: {:.9f}".format(accuracy.eval(
                            #{tf_train_dataset: train_dataset, tf_train_labels: train_labels, keep_prob: 0.5})))
                        #print("Validation accuracy: {:.9f}".format(accuracy.eval(
                            #{tf_train_dataset: valid_dataset, tf_train_labels: valid_labels, keep_prob: 1.0})))


            saver.save(sess, checkpoint)
            if epoch % display_step == 0:
                print "Epoche:", '%04d' % (epoch + 1), "of", '%04d' % (hm_epochs), "average cost=", "{:.9f}".format(avg_cost)

            with open(tf_log, 'a') as f:
                f.write(str(epoch) + '\n')



        feature_sets = []
        labels = []
        zaehler = 0

        with tf.gfile.Open(csv_file2, 'rb') as gc_file:
            lines2 = gc_file.readlines()

            for zeile in lines2:
                try:
                    features = list(eval(zeile.split('::')[0]))
                    label = list(eval(zeile.split('::')[1]))
                    feature_sets.append(features)
                    labels.append(label)
                    zaehler += 1
                except:
                    pass

        test_x = np.array(feature_sets)
        test_datasets = test_x

        test_y = np.array(labels)
        test_labels = test_y
        print('Test Set', test_datasets.shape, test_labels.shape)

        writer.flush()

        print 'Test Accuracy:', accuracy.eval(
            {tf_train_dataset: test_datasets, tf_train_labels: test_labels, keep_prob: 1.0})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
        '--train-file',
        help='GCS or local paths to training data',
        required=True
    )
    parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models',
        required=True
    )
    parser.add_argument('--csv-file',
                        help='csv file',
                        required=True)
    parser.add_argument('--csv-file2',
                        help='csv file2',
                        required=True)
    parser.add_argument('--checkpoint',
                        help='checkpointfile',
                        required=True)
    parser.add_argument('--logs',
                        help='logfile',
                        required=True)
    args = parser.parse_args()
    arguments = args.__dict__

    trainDNN(**arguments)