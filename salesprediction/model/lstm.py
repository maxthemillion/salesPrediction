from numpy.random import seed
seed(42)
from tensorflow import set_random_seed
set_random_seed(42)

 
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
from keras.models import load_model


import pickle
import os
import csv

import matplotlib.pyplot as plt
import plotly.graph_objects as go
#import datetime
from datetime import datetime
from dateutil.relativedelta import relativedelta



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
MAX_YEAR = 1 #2013 2014 2015
MAX_MONTH = 2 # 4 7 8 9 10, weil shape(4,5147)
MAX_SHOP = len(TEST_SHOPS)

LENGTH = MAX_SHOP + MAX_ITEM + MAX_MONTH + 1 + 1 + 1
MAXLEN = 4 # 4 months
STEP = 1


CNT_SCALER = StandardScaler()
CNT_SCALER.fit(TRAIN.item_cnt_day.as_matrix().reshape(-1, 1))

BATCH_SIZE = 256
EPOCHS = 10



def run():
    (x_train_o, x_val_o, x_test_o, y_train, y_val) = _loadData()

    (shop_dm, item_dm, month_dm) = _oneHotEncoding()

    x_train = _vectorize(x_train_o, shop_dm, item_dm, month_dm)
    x_val = _vectorize(x_val_o, shop_dm, item_dm, month_dm)
    x_test = _vectorize(x_test_o, shop_dm, item_dm, month_dm)

    model = _buildModel()#Modelname
    model = _fit(model, x_train, y_train)
    (predict_train, predict_val,duration) = _predict(model, x_train, x_val)#

    train_score = _evaluate(y_train, predict_train)#
    val_score = _evaluate(y_val, predict_val)#

    _save(model, val_score)
    _submit(model, x_test, x_test_o, val_score)
    #_CreateTable(MAX_YEAR,BATCH_SIZE,EPOCHS,train_score,val_score,duration)# at first the table have to create
    _AddValues(MAX_YEAR,BATCH_SIZE,EPOCHS,train_score,val_score,duration) #if the table existed, then you allow to add new value in the list. Please donot use both function (_Createtable und _AddValues) at the same time  
    T_list = _zeitspanne()
    zeit_list = _zeitaxis()
    list_2013 = _verkaufszahlen_2013()
    list_2014 = _verkaufszahlen_2014()
    list_2015 = _verkaufszahlen_2015()
    y_VH_112015 = _Vorhersage_112015()
    _graphischeVisualisierung(T_list,list_2015,list_2014,list_2013,y_VH_112015)
    y_value = _Y_werteberechen()
    _graphischeVisualisierung_gesamt(zeit_list,y_value,y_VH_112015)


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


    print('\n')
    return x


def _oneHotEncoding():
    shop_le = preprocessing.LabelEncoder()
    shop_le.fit(TEST_SHOPS)
    shop_dm = dict(zip(TEST_SHOPS, shop_le.transform(TEST_SHOPS))) #shop_dm bekommt die Listen TEST_SHOPS, shop_le.transform(TEST_SHOPS

    item_le = preprocessing.LabelEncoder()
    item_le.fit(TEST_ITEMS)
    item_dm = dict(zip(TEST_ITEMS, item_le.transform(TEST_ITEMS))) #item_dm bekommt die Listen TEST_ITEMS, item_le.transform(TEST_ITEMS)

    month_le = preprocessing.LabelEncoder()
    month_le.fit(range(7,11)) #Monat Juli bis Oktober

    month_dm = dict(zip(range(7,11), month_le.transform(range(7,11))))

    #cat_le = preprocessing.LabelEncoder()
    #cat_le.fit(item_cats.item_category_id)
    #cat_dm = dict(zip(item_cats.item_category_id.unique(), cat_le.transform(item_cats.item_category_id.unique())))
    # das ist kein OneHotEncoding, sondern die Informationen aus aus der Liste wird zu einem Diktionararray umgewandelt
    return (shop_dm, item_dm, month_dm)


def _buildModel():
    modelName = config.getModelPath()

    if modelName == '':
        # build the model: a single LSTM
        print('Building model...')
        model = Sequential()
        model.add(LSTM(16, input_shape=(MAXLEN, LENGTH))) 
        model.add(LSTM(8, input_shape=(MAXLEN, LENGTH))) 
        model.add(Dense(4, activation='relu'))
        model.add(Dense(1, activation='relu'))

        model.compile(loss='mean_squared_error', optimizer=RMSprop(lr=0.0035, rho=0.9, epsilon=None, decay=0.0))

    else:
        model = load_model(modelName)

    print('\n')
    return model


def _fit(model, x_train, y_train):
    print('Fitting model...')
    model.fit(x_train, y_train, BATCH_SIZE, EPOCHS)

    print('\n')
    return model


def _save(model, val_score):
    print('Saving model ...')

    timestr = time.strftime("%Y%m%d-%H%M")
    model.save(os.path.join(config.MODELS_PATH, (str(round(val_score,2)) + 'RMSE_' + timestr +'_model.h5')))

    print('Sucessfully saved model to folder.')
    print('\n')


def _predict(model, x_train, x_val):
    #make predictions on train and validation set
    print('Start predicting...')
    start = time.time()

    predict_train = model.predict(x_train)
    predict_val = model.predict(x_val)

    duration = time.time() - start
    print('Prediction took ' + str(round(duration, 2)))

    return (predict_train, predict_val, duration)

def _evaluate(y, y_hat):
    print('Start evaluating...')
    start = time.time()

    y_hat_inverse = CNT_SCALER.inverse_transform(y_hat)
    y_inverse = CNT_SCALER.inverse_transform(y)
   
    score = math.sqrt(mean_squared_error(y_hat_inverse, y_inverse))
    print('Score: %.2f RMSE' % (score))

    duration = time.time() - start
    print('Evaluating took ' + str(round(duration, 2)))
    print('\n')
    return score


def _submit(model, x_test, x_test_o, val_score):
    print('Predicting on the test set...')

    predict_test = model.predict(x_test)
    predict_test = CNT_SCALER.inverse_transform(predict_test)

    test = TEST.set_index(['shop_id', 'item_id'])
    test['item_cnt_month'] = 0

    for index, sentence in enumerate(x_test_o):
        (shop_id, item_id) = (sentence[0]['shop_id'], sentence[0]['item_id'])
        test.loc[(shop_id, item_id)]['item_cnt_month'] = predict_test[index]

    test = test.reset_index().drop(['shop_id', 'item_id'], axis=1)

    timestr = time.strftime("%Y%m%d-%H%M")
    test.to_csv(os.path.join(config.SUBMISSIONS_PATH, (str(round(val_score,2))+'RMSE_' + timestr +'_submission.csv')), index=False)

    print('Successfully saved predictions to folder. Happy submitting!')
    print('\n')

# create the csv table 
def _CreateTable(a,b,c,d,e,f):
    MY =[]
    BS= []
    E = []
    ST =[]
    SR = []
    RZ = []
    MY.append(a)
    BS.append(b)
    E.append(c)
    ST.append(d)
    SR.append(e)
    RZ.append(f)
    dict = {'MAX_Year': MY,'Batch_Size': BS,'Epochs': E,'Score(TraningSet)': ST,'Score(RealSet)': SR,'Rechenzeit':RZ}  
    df = pd.DataFrame(dict) 
    df.to_csv('SaveValue.csv',index = False )  

# add new Value into the existing csv.table
def _AddValues(MY,BS,E,ST,SR,RZ):
    A = [MY,BS,E,ST,SR,RZ]
    df = pd.DataFrame(columns =  ['MAX_Year','Batch_Size','Epochs','Score(TraningSet)','Score(RealSet)','Rechenzeit'])
    df = pd.DataFrame(np.array([A]),
    columns=['MAX_Year','Batch_Size','Epochs','Score(TraningSet)','Score(RealSet)','Rechenzeit']).append(df, ignore_index=False)
    df.to_csv('SaveValue.csv', mode='a', header=False,index = False)

#create the time list from august to october
def _zeitspanne():

    startDate = '01-08'
    endDate = '01-12'
    datum_list = []
    cur_date = start = datetime.strptime(startDate, '%d-%m').date()
    end = datetime.strptime(endDate, '%d-%m').date()
    step = relativedelta(months=1)
    while cur_date < end:
        datum_list.append(cur_date)
        cur_date = cur_date + step     
    return datum_list

#create the time list from jan 2013 to nov 2015
def _zeitaxis():
    
    startDate = '01-2013'
    endDate = '12-2015'
    datum_list = []
    cur_date = start = datetime.strptime(startDate, '%m-%Y').date()
    end = datetime.strptime(endDate, '%m-%Y').date()
    step = relativedelta(months=1)
    while cur_date < end:
        datum_list.append(cur_date)
        cur_date = cur_date + step     
    return datum_list

#create the list of the Sales of 2013 from august to nov
def _verkaufszahlen_2013():
    df = pd.read_csv('C:/Users/yanga/Desktop/challenge/salesPrediction/data/sales_train_v2.csv',delimiter = ',')
    df['date']=pd.to_datetime(df['date'] ,errors = 'coerce', format = '%d.%m.%Y').dt.strftime("%m-%Y")
    VKZ_08_2013 = df[df['date']== '08-2013']
    VKZ_08_2013_sum = VKZ_08_2013['item_cnt_day'].sum()
    VKZ_09_2013 = df[df['date']=='09-2013']
    VKZ_09_2013_sum = VKZ_09_2013['item_cnt_day'].sum()
    VKZ_10_2013 = df[df['date']=='10-2013']
    VKZ_10_2013_sum = VKZ_10_2013['item_cnt_day'].sum()
    VKZ_11_2013 = df[df['date']=='11-2013']
    VKZ_11_2013_sum = VKZ_11_2013['item_cnt_day'].sum()
    return [VKZ_08_2013_sum,VKZ_09_2013_sum,VKZ_10_2013_sum,VKZ_11_2013_sum]

#create the list of the Sales of 2014 from august to nov
def _verkaufszahlen_2014():
    df = pd.read_csv('C:/Users/yanga/Desktop/challenge/salesPrediction/data/sales_train_v2.csv',delimiter = ',')
    df['date']=pd.to_datetime(df['date'] ,errors = 'coerce', format = '%d.%m.%Y').dt.strftime("%m-%Y")
    VKZ_08_2014 = df[df['date']== '08-2014']
    VKZ_08_2014_sum = VKZ_08_2014['item_cnt_day'].sum()
    VKZ_09_2014 = df[df['date']=='09-2014']
    VKZ_09_2014_sum = VKZ_09_2014['item_cnt_day'].sum()
    VKZ_10_2014 = df[df['date']=='10-2014']
    VKZ_10_2014_sum = VKZ_10_2014['item_cnt_day'].sum()
    VKZ_11_2014 = df[df['date']=='11-2014']
    VKZ_11_2014_sum = VKZ_11_2014['item_cnt_day'].sum()
    return [VKZ_08_2014_sum,VKZ_09_2014_sum,VKZ_10_2014_sum,VKZ_11_2014_sum]
#create the list of the Sales of 2015 from august to nov
def _verkaufszahlen_2015():
    df = pd.read_csv('C:/Users/yanga/Desktop/challenge/salesPrediction/data/sales_train_v2.csv',delimiter = ',')
    df['date']=pd.to_datetime(df['date'] ,errors = 'coerce', format = '%d.%m.%Y').dt.strftime("%m-%Y")
    VKZ_08_2015 = df[df['date']== '08-2015']
    VKZ_08_2015_sum = VKZ_08_2015['item_cnt_day'].sum()
    VKZ_09_2015 = df[df['date']=='09-2015']
    VKZ_09_2015_sum = VKZ_09_2015['item_cnt_day'].sum()
    VKZ_10_2015 = df[df['date']=='10-2015']
    VKZ_10_2015_sum = VKZ_10_2015['item_cnt_day'].sum()
    return [VKZ_08_2015_sum,VKZ_09_2015_sum,VKZ_10_2015_sum]

#create the visualization from aug to nov
def _graphischeVisualisierung(x,y,y_2014,y_2013,y1):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = x,y = y,name = "real 08-2015->10-2015",line_color = "blue"))
    fig.add_trace(go.Scatter(x=x[-1:], y=y1,mode='markers',name='Vorhersage für Nov 2015'))
    fig.add_trace(go.Scatter(x = x,y = y_2014,name = "real 08-2014->11-2014",line_color = "green"))
    fig.add_trace(go.Scatter(x = x,y = y_2013,name = "real 08-2013->11-2013",line_color = "yellow"))
    fig.update_layout(title_text= "Vorhersage für den 11-2015 (Produkt-Store-Kombinationen)",
    xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text="Datum (Aug. bis Okt.)",font=dict(family="Courier New, monospace",size=18,color="#7f7f7f"))),
    yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text="Anzahl der verkauften Waren pro Monat",font=dict(family="Courier New, monospace",size=18,color="#7f7f7f"))))
    fig.show()
    #export the graph
    #fig.write_image("fig1.png")

#create the visualization from jan 2013 to okc 2015
def _graphischeVisualisierung_gesamt(x,y,y1):
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = x,y = y,mode='lines+markers',line_color = "blue",name="Verkaufszahlen aus der Vergangenheit"))
    fig.add_trace(go.Scatter(x=x[-1:], y=y1,mode='markers',name='Vorhersage für Nov 2015'))
    fig.update_layout(title_text= "Vorhersage für den 11-2015 (Produkt-Store-Kombinationen)",
    xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text="Datum (Jan 2013 bis Okt 2015)",font=dict(family="Courier New, monospace",size=18,color="#7f7f7f"))),
    yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text="Anzahl der verkauften Waren pro Monat",font=dict(family="Courier New, monospace",size=18,color="#7f7f7f"))))
    fig.show()
    #export the graph
    #fig.write_image("fig2.png")

#create the list of sell good from jan 2013 to okc 2015
def _Y_werteberechen():
    df = pd.read_csv('C:/Users/yanga/Desktop/challenge/salesPrediction/data/sales_train_v2.csv',delimiter = ',')
    df1 = df.sort_values('date')
    df1['date'] = pd.to_datetime(df1['date'] ,errors = 'coerce', format = '%d.%m.%Y').dt.strftime("%Y-%m")
    df2 = df1.groupby(['date']).sum().sort_values('date',ascending=True)
    return df2['item_cnt_day']

#create the prognosis of nov 2015
def _Vorhersage_112015():
    df3 = pd.read_csv('C:/Users/yanga/Desktop/challenge/salesPrediction/submissions/0.3RMSE_20190715-1716_submission.csv')
    y1 = []
    y1.append(df3['item_cnt_month'].sum())
    return y1
        



        
        
                        
        
        