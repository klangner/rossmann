import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.ensemble import RandomForestRegressor
import math


DUMMIES = ['StoreType', 'DayOfWeek', 'month', 'PromoInterval', 'Assortment', 'StateHoliday']


def load_train():
    train = pd.read_csv("../data/train.csv", dtype={'StateHoliday': str}, parse_dates=["Date"])
    train = train.loc[train.Sales > 0]
    store = pd.read_csv("../data/store.csv")
    return pd.merge(train, store, 'left', on='Store', copy=False)


def build_features(df):
    df.fillna(0, inplace=True)
    df.loc[df.Open.isnull(), 'Open'] = 1
    # Base features
    features = ['Store', 'CompetitionDistance', 'Promo', 'Promo2', 'SchoolHoliday']
    # Date & Time
    features.extend(['year', 'month', 'week', 'day', 'DayOfWeek'])
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['week'] = df['Date'].dt.weekofyear
    df['day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    # Calculate time competition open time in months
    # https://www.kaggle.com/ananya77041/rossmann-store-sales/randomforestpython/code
    features.append('CompetitionOpen')
    df['CompetitionOpen'] = 12 * (df.year - df.CompetitionOpenSinceYear) + (df.month - df.CompetitionOpenSinceMonth)
    # Promo open time in months
    features.append('PromoOpen')
    df['PromoOpen'] = 12 * (df.year - df.Promo2SinceYear) + (df.week - df.Promo2SinceWeek) / 4.0
    df['PromoOpen'] = df.PromoOpen.apply(lambda x: x if x > 0 else 0)
    df.loc[df.Promo2SinceYear == 0, 'PromoOpen'] = 0
    # Indicate that sales on that day are in promo interval
    features.append('IsPromoMonth')
    month2str = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                 7: 'Jul', 8: 'Aug', 9: 'Sept', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    df['monthStr'] = df.month.map(month2str)
    df.loc[df.PromoInterval == 0, 'PromoInterval'] = ''
    df['IsPromoMonth'] = 0
    for interval in df.PromoInterval.unique():
        if interval != '':
            for month in interval.split(','):
                df.loc[(df.monthStr == month) & (df.PromoInterval == interval), 'IsPromoMonth'] = 1
    return df[features]


def cross_validate():
    df = load_train()
    features = build_features(df)
    sales = df['Sales']
    (X, X2, y, y2) = cross_validation.train_test_split(features, sales)
    regressor = RandomForestRegressor(n_jobs=-1, n_estimators=15)
    model = regressor.fit(X, np.log(y))
    y = np.power(math.e, model.predict(X2))
    spe = ((y2 - y) / y2) ** 2.0
    rmspe = np.sqrt(spe.sum() / len(spe))
    print("RMSPE=%f" % rmspe)


def load_test():
    test = pd.read_csv("../data/test.csv", dtype={'StateHoliday': str}, parse_dates=["Date"])
    store = pd.read_csv("../data/store.csv")
    return pd.merge(test, store, 'left', on='Store', copy=False)


def build_solution():
    train = load_train()
    X = build_features(train)
    y = train['Sales']
    regressor = RandomForestRegressor()
    model = regressor.fit(X, np.log(y))

    test = load_test()
    X2 = build_features(test)
    test['Sales'] = np.power(math.e, model.predict(X2))
    test.loc[test.Open == 0, 'Sales'] = 0
    test[['Id', 'Sales']].astype(int).to_csv('../data/solution.csv', index=False)


cross_validate()
# build_solution()
print('Done.')

