import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import cross_validation


COLUMNS = ['StoreType_a', 'StoreType_b', 'StoreType_c', 'StoreType_d',
           'DayOfWeek_1', 'DayOfWeek_2', 'DayOfWeek_3', 'DayOfWeek_4', 'DayOfWeek_5', 'DayOfWeek_6', 'DayOfWeek_7',
           'Promo',
           'month_1', 'month_2', 'month_3', 'month_4', 'month_5', 'month_6', 'month_7', 'month_8', 'month_9',
           'month_10', 'month_11', 'month_12']


def load_train():
    train = pd.read_csv("../data/train.csv", dtype={'StateHoliday': str}, parse_dates=["Date"])
    train = train.loc[train.Sales > 0]
    train['month'] = train['Date'].dt.month
    train['day'] = train['Date'].dt.day
    store = pd.read_csv("../data/store.csv")
    df = pd.merge(train, store, 'left', on='Store')
    df = pd.get_dummies(df, columns=['StoreType', 'DayOfWeek', 'month'])
    return df[:10]


def build_model(x_placeholder):
    # Try to find values for W and b that compute y_data = W * x_data + b
    W = tf.Variable(tf.random_uniform([len(COLUMNS)], -1.0, 1.0))
    b = tf.Variable(tf.zeros([1]))
    return tf.matmul(x_placeholder, W) + b


def cost(model, expected):
    # Minimize the mean squared errors.
    loss = tf.reduce_mean(tf.square(model - expected))
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    return optimizer.minimize(loss)


def cross_validate():
    (cv_train, cv_test) = cross_validation.train_test_split(load_train())
    x_placeholder = tf.placeholder("float", shape=[None, len(COLUMNS)])
    y_placeholder = tf.placeholder("float", shape=[None])
    model = build_model(x_placeholder)
    init = tf.initialize_all_variables()
    # Launch the graph.
    sess = tf.Session()
    sess.run(init)
    loss = cost(model, y_placeholder)
    for _ in range(2):
        sess.run(loss, feed_dict={x_placeholder: cv_train[COLUMNS], y_placeholder: cv_train['Sales']})

    y = sess.run(model, feed_dict={x_placeholder: cv_test[COLUMNS]})
    spe = ((cv_test['Sales'] - y) / cv_test['Sales']) ** 2.0
    rmspe = np.sqrt(spe.sum() / len(spe))
    print("RMSPE=%f" % rmspe)


cross_validate()

