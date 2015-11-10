import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn import tree

COLUMNS = ['Store', 'DayOfWeek', 'Promo', 'month']


def load_train():
    train = pd.read_csv("../data/train.csv", dtype={'StateHoliday': str}, parse_dates=["Date"])
    train = train.loc[train.Sales > 0]
    train['month'] = train['Date'].dt.month
    train['day'] = train['Date'].dt.day
    return train


def build_model():
    train = load_train()
    dt = tree.DecisionTreeClassifier()
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
    dt = tree.DecisionTreeRegressor()
    model = dt.fit(cv_train[COLUMNS], cv_train['Sales'])
    y = model.predict(cv_test[COLUMNS])
    spe = ((cv_test['Sales'] - y) / cv_test['Sales']) ** 2.0
    rmspe = np.sqrt(spe.sum() / len(spe))
    print("RMSPE=%f" % rmspe)


cross_validate()
