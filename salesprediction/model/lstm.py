from ..config import config
from sklearn import preprocessing

import numpy as np
import pandas as pd

import math
from math import ceil

import time

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop

import pickle
import os

TEST = pd.read_csv(os.path.join(config.DATA_PATH, './test.csv'))
TRAIN = pd.read_csv(os.path.join(config.DATA_PATH, './sales_train_v2.csv'))
ITEM_CATS = pd.read_csv(os.path.join(config.DATA_PATH,'./item_categories.csv'))

TEST_SHOPS = TEST.shop_id.unique()
TRAIN = TRAIN[TRAIN.shop_id.isin(TEST_SHOPS)]
TEST_ITEMS = TEST.item_id.unique()
TRAIN = TRAIN[TRAIN.item_id.isin(TEST_ITEMS)]

MAX_BLOCK_NUM = TRAIN.date_block_num.max()
MAX_ITEM = len(TEST_ITEMS)
MAX_CAT = len(ITEM_CATS)
MAX_YEAR = 3
MAX_MONTH = 2 # 7 8 9 10
MAX_SHOP = len(TEST_SHOPS)

LENGTH = MAX_SHOP + MAX_ITEM + MAX_MONTH + 1 + 1 + 1
MAXLEN = 4 # 4 months
STEP = 1

def run():
    (x_train_o, x_val_o, x_test_o, y_train, y_val) = _loadData()

    (shop_dm, item_dm, month_dm) = _oneHotEncoding()

    x_train = _vectorize(x_train_o, shop_dm, item_dm, month_dm)
    x_val = _vectorize(x_val_o, shop_dm, item_dm, month_dm)
    x_test = _vectorize(x_test_o, shop_dm, item_dm, month_dm)

    model = _buildModel()
    model = _fit(model, x_train, y_train)
    (predict_train, predict_val) = _predict(model, x_train, x_val)

    train_score = _evaluate(y_train, predict_train)
    val_score = _evaluate(y_val, predict_val)

    return True


def _loadData():
    with open(os.path.join(config.DATA_PATH, './x_train_o'), 'rb') as file:
        x_train_o = pickle.load(file)

    with open(os.path.join(config.DATA_PATH, './x_val_o'), 'rb') as file:
        x_val_o = pickle.load(file)
        
    with open(os.path.join(config.DATA_PATH, './x_test_o'), 'rb') as file:
        x_test_o = pickle.load(file)
        
    with open(os.path.join(config.DATA_PATH, './y_train'), 'rb') as file:
        y_train = pickle.load(file)
        
    with open(os.path.join(config.DATA_PATH, './y_val'), 'rb') as file:
        y_val = pickle.load(file)

    return (x_train_o, x_val_o, x_test_o, y_train, y_val)


def _vectorize(inp, shop_dm, item_dm, month_dm):
    print('Vectorization...')   
    x = np.zeros((len(inp), MAXLEN, LENGTH), dtype=np.float32)
    for i, sentence in enumerate(inp):
        for t, char in enumerate(sentence):            
            x[i][t][ shop_dm[char['shop_id']] ] = 1        
            x[i][t][ MAX_SHOP + item_dm[char['item_id']] ] = 1
            x[i][t][ MAX_SHOP + MAX_ITEM + month_dm[char['month']] ] = 1
            x[i][t][ MAX_SHOP + MAX_ITEM + MAX_MONTH + 1 ] = char['item_price']
            x[i][t][ MAX_SHOP + MAX_ITEM + MAX_MONTH + 1 + 1] = char['item_cnt_day']    
    return x


def _oneHotEncoding():
    shop_le = preprocessing.LabelEncoder()
    shop_le.fit(TEST_SHOPS)
    shop_dm = dict(zip(TEST_SHOPS, shop_le.transform(TEST_SHOPS)))

    item_le = preprocessing.LabelEncoder()
    item_le.fit(TEST_ITEMS)
    item_dm = dict(zip(TEST_ITEMS, item_le.transform(TEST_ITEMS)))

    month_le = preprocessing.LabelEncoder()
    month_le.fit(range(7,11))
    month_dm = dict(zip(range(7,11), month_le.transform(range(7,11))))

    #cat_le = preprocessing.LabelEncoder()
    #cat_le.fit(item_cats.item_category_id)
    #cat_dm = dict(zip(item_cats.item_category_id.unique(), cat_le.transform(item_cats.item_category_id.unique())))
    return (shop_dm, item_dm, month_dm)

def _buildModel():
    # build the model: a single LSTM
    print('Build model...')
    model = Sequential()
    model.add(LSTM(32, input_shape=(MAXLEN, LENGTH)))
    model.add(Dense(1, activation='relu'))

    optimizer = RMSprop(lr=0.005)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model


def _fit(model, x_train, y_train):
    batch_size = 128
    epochs = 13
    model.fit(x_train, y_train, batch_size, epochs)
    return model


def _predict(model, x_train, x_val):
    #make predictions on train and validation set
    predict_train = model.predict(x_train)
    predict_val = model.predict(x_val)
    return (predict_train, predict_val)


def _evaluate(y, y_hat):
    y_hat_inverse = cnt_scaler.inverse_transform(predict_train)
    y_inverse = cnt_scaler.inverse_transform(y_train)
    score = math.sqrt(mean_squared_error(y_hat_inverse, y_inverse))
    print('Score: %.2f RMSE' % (score))
    return score


def _evaluate(predict_train, y_train, predict_val, y_val):    
    scaler = StandardScaler()
    cnt_scaler = StandardScaler()

    scaler.fit(TRAIN.item_price.as_matrix().reshape(-1, 1))
    cnt_scaler.fit(TRAIN.item_cnt_day.as_matrix().reshape(-1, 1))

    #invert predictions
    predict_train = cnt_scaler.inverse_transform(predict_train)
    y_train = cnt_scaler.inverse_transform(y_train)
    predict_val = cnt_scaler.inverse_transform(predict_val)
    y_val = cnt_scaler.inverse_transform(y_val)
    
    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(predict_train, y_train))
    print('Train Score: %.2f RMSE' % (trainScore))

    valScore = math.sqrt(mean_squared_error(predict_val, y_val))
    print('Test Score: %.2f RMSE' % (valScore))

    # TODO: Why is there no test score?
    return (trainScore, valScore)


def _submit(model, x_test, x_test_o):
    predict_test = model.predict(x_test)
    predict_test = cnt_scaler.inverse_transform(predict_test)

    test = TEST.set_index(['shop_id', 'item_id'])
    test['item_cnt_month'] = 0

    for index, sentence in enumerate(x_test_o):
        (shop_id, item_id) = (sentence[0]['shop_id'], sentence[0]['item_id'])
        test.loc[(shop_id, item_id)]['item_cnt_month'] = predict_test[index]

    test = test.reset_index().drop(['shop_id', 'item_id'], axis=1)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    test.to_csv(os.path.join(config.EXPORT_PATH, (timestr+'_submission.csv')), index=False)