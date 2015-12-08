#
# TensorFlow based solution
#

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn import cross_validation


FEATURES_COUNT = 7
HIDDEN_NEURON_COUNT = 15
COLUMNS = ['Store', 'DayOfWeek', 'Promo', 'month']


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def load_cross_validation():
    df = pd.read_csv('../data/train.csv', dtype={'StateHoliday': str}, parse_dates=["Date"])
    (cv_train, cv_test) = cross_validation.train_test_split(df, train_size=0.75)
    train_sales = cv_train['Sales']
    train_df = pd.get_dummies(cv_train['DayOfWeek'])
    test_sales = cv_test['Sales']
    test_df = pd.get_dummies(cv_test['DayOfWeek'])
    return train_df, train_sales, test_df, test_sales


def build_model(x_placeholder):
    """ Create network with single hidden layer and single output neuron. """
    # Layer 2
    w2 = weight_variable([FEATURES_COUNT, HIDDEN_NEURON_COUNT])
    b2 = bias_variable([HIDDEN_NEURON_COUNT])
    l2 = tf.matmul(x_placeholder, w2) + b2
    # Layer 3
    w3 = weight_variable([HIDDEN_NEURON_COUNT, 1])
    b3 = bias_variable([1])
    l3 = tf.matmul(l2, w3) + b3
    return l3


def calculate_score(expected, response):
    spe = tf.pow((response-expected) / expected, 2)
    return tf.sqrt(spe)


def train(session, model, x_placeholder, y_placeholder, df_train, df_sales):
    """ Train network on given data """
    cross_entropy = calculate_score(y_placeholder, model)
    accuracy = calculate_score(y_placeholder, model)
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
    for i in range(200):
        (train_a, _, label_a, _) = cross_validation.train_test_split(df_train, df_sales, train_size=100)
        session.run(train_step, feed_dict={x_placeholder: train_a, y_placeholder: label_a})
        if i % 10 == 0:
            train_accuracy = session.run(accuracy,
                                         feed_dict={x_placeholder: train_a, y_placeholder: label_a})
            print "step %d, training accuracy %g" % (i, train_accuracy)
    return model


def main():
    x_placeholder = tf.placeholder("float", shape=[None, 7])
    y_placeholder = tf.placeholder("float", shape=[None])
    train_df, train_sales, test_df, test_sales = load_cross_validation()
    model = build_model(x_placeholder)
    session = tf.Session()
    init = tf.initialize_all_variables()
    session.run(init)
    model = train(session, model, x_placeholder, y_placeholder, train_df, train_sales)
    y = session.run(model, feed_dict={x_placeholder: test_df})
    spe = ((test_sales - y) / test_sales) ** 2.0
    score = np.sqrt(spe.sum() / len(spe))
    print("Final score %f" % score)


main()
