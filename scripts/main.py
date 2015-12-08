import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn import linear_model


COLUMNS2 = ['StoreType_a', 'StoreType_b', 'StoreType_c', 'StoreType_d',
           'DayOfWeek_1', 'DayOfWeek_2', 'DayOfWeek_3', 'DayOfWeek_4', 'DayOfWeek_5', 'DayOfWeek_6', 'DayOfWeek_7',
           'Promo',
           'month_1', 'month_2', 'month_3', 'month_4', 'month_5', 'month_6', 'month_7', 'month_8', 'month_9',
           'month_10', 'month_11', 'month_12']

COLUMNS = ['DayOfWeek_1', 'DayOfWeek_2', 'DayOfWeek_3', 'DayOfWeek_4', 'DayOfWeek_5', 'DayOfWeek_6', 'DayOfWeek_7',
           'Promo']


def load_train():
    train = pd.read_csv("../data/train.csv", dtype={'StateHoliday': str}, parse_dates=["Date"])
    train = train.loc[train.Sales > 0]
    train['month'] = train['Date'].dt.month
    train['day'] = train['Date'].dt.day
    store = pd.read_csv("../data/store.csv")
    df = pd.merge(train, store, 'left', on='Store')
    df = pd.get_dummies(df, columns=['StoreType', 'DayOfWeek', 'month'])
    return df


def build_model():
    train = load_train()
    dt = linear_model.LinearRegression()
    return dt.fit(train[COLUMNS], train['Sales'])


def build_solution(model):
    test = pd.read_csv("../data/test.csv", dtype={'StateHoliday': str}, parse_dates=["Date"])
    test.loc[test.Open.isnull(), 'Open'] = 1
    test['year'] = test['Date'].dt.year
    test['month'] = test['Date'].dt.month
    test['day'] = test['Date'].dt.day
    test['Sales'] = model.predict(test[COLUMNS])
    test.loc[test.Open == 0, 'Sales'] = 0
    test[['Id', 'Sales']].astype(int).to_csv('../data/solution.csv', index=False)


def cross_validate():
    (cv_train, cv_test) = cross_validation.train_test_split(load_train())
    dt = linear_model.LinearRegression()
    model = dt.fit(cv_train[COLUMNS], cv_train['Sales'])
    y = model.predict(cv_test[COLUMNS])
    spe = ((cv_test['Sales'] - y) / cv_test['Sales']) ** 2.0
    rmspe = np.sqrt(spe.sum() / len(spe))
    print("RMSPE=%f" % rmspe)


def solve():
    model = build_model()
    build_solution(model)


cross_validate()
# solve()
print('Done.')

