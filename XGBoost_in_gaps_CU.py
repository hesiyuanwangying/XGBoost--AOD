import pandas as pd
import numpy as np
import xgboost as xgb
from scipy import stats
import pprint
# import matplotlib.pyplot as plt
# import time


def data_preprocessing():
    '''read the type of xls file to array type,then map the label to specific number.training_data
    and testing_data had already been respectively stored in 4-1.xls and 4-2.xls. Besides, 4-3.xls stored the data
    which is lacked AOD data but have corresponding meteorological data '''
    drop_columns_all = 'springaod'
    targetName = 'state'
    cols = ['dxzf', 'pjdw', 'pjfs', 'pjqw', 'pjsd', 'qts',
            'rzsj', 'zdfs']
    # transform the data to specific labels
    value_to_lable = {0.0001: 0,
                      0.25: 1,
                      0.5: 2,
                      1: 3,
                      1.5: 4,
                      1.95: 5}
    # read data from different files and transform them to suitable type
    df = pd.read_excel(r'.\4-1.xls')
    df[targetName] = df[drop_columns_all].map(value_to_lable)
    X_train = df[cols]
    X_train = np.array(X_train)
    Y_train = df[targetName]
    Y_train = np.array(Y_train)

    df1 = pd.read_excel(r'.\4-2.xls')
    df1[targetName] = df1[drop_columns_all].map(value_to_lable)
    X_test = df1[cols]
    X_test = np.array(X_test)
    Y_test = df1[targetName]
    Y_test = np.array(Y_test)
    return X_train, X_test, Y_train, Y_test


X_train, X_test, Y_train, Y_test = data_preprocessing()
# check train and test data feature intuitively by hist graph and statistical analysis
Y_train_pd = pd.Series(Y_train)
Y_train_pd.plot.hist(grid=True, bins=20, rwidth=0.9,
                   color='#607c8e')
Y_train_pd.describe()

Y_test_pd = pd.Series(Y_test)
Y_test_pd.plot.hist(grid=True, bins=20, rwidth=0.9,
                   color='#607c8e')
Y_test_pd.describe()

pprint.pprint(stats.describe(X_train),width=1)


def my_error(y_pred, dtrain):
    y_true = dtrain.get_label()
    print(y_pred)
    print(y_true)
    tp = sum([int(i == 1 and j == 1) for i, j in zip(y_pred, y_true)])
    precision = float(tp)/sum(y_pred)
    recall = float(tp)/sum(y_true)
    return 'f1-score', 2 * (precision*recall/(precision+recall))


def all_err_rate(result):
    error_rate = np.zeros(6)
    for i in range(0, 6):
        row = result.loc[result['test_label'] == i]
        error_rate[i] = np.sum(row['test_label'] != row['test_pred'])*1.0/row.shape[0]
    return error_rate


def xgboost_model(train_weight, test_weight, train_label, test_label):
    # date to Dmatrix
    xgb_train = xgb.DMatrix(train_weight, train_label)
    xgb_test = xgb.DMatrix(test_weight, test_label)
    # set parameters
    param = {'max_depth': 20, 'eta': 0.3, 'eval_metric': 'merror', 'silent': 1, 'objective': 'multi:softmax',
             'num_class': 6}
    watchlist = [(xgb_train, 'train'), (xgb_test, 'test')]
    num_ronud = 50
    # train
    bst = xgb.train(param, xgb_train, num_ronud, watchlist)
    # predict
    pred_label = bst.predict(xgb_test)
    pred_label_train = bst.predict(xgb_train)
    error_rate = np.sum(pred_label != test_label) * 1.0 / test_label.shape[0]
    print('xgboost model accuracy: %f' % (1 - error_rate))
    result = pd.DataFrame(pred_label, columns=['test_pred'])
    result['test_pred'] = pd.DataFrame(pred_label)
    result['test_label'] = pd.DataFrame(test_label)
    result['train_pred'] = pd.DataFrame(pred_label_train)
    result['train_label'] = pd.DataFrame(train_label)

    result.to_csv('./out.csv')
    return result


result = xgboost_model(X_train, X_test, Y_train, Y_test)

print(1-all_err_rate(result))
