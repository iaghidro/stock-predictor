#from fastai.structured import *
#from fastai.column_data import *
#np.set_printoptions(threshold=50, edgeitems=20)

import pandas as pd
import numpy as np

from stockPredictor import StockPredictor

data = {
    'Close': [1, 25, 6, 34, 6], 
    'Open': [3, 4, 78, 56, 7],
    'Timestamp': [1325317920,1325317980,1325318040,1325318100,1325318160]
} 
missingData = {
    'Close': [1, np.nan, 6, 34, 6], 
    'Open': [np.nan, -np.inf, 78, 56, 7]
}  
backfilledData = {
    'Close': [1, 6, 6, 34, 6], 
    'Open': [78, 78, 78, 56, 7]
}    
testRecordsCount=2
trainRecordsCount=3
index = 'Timestamp'

def createPredictor():
    df = pd.DataFrame(data=data)
    return StockPredictor(df, index)
    
def test_constructor():
    df = pd.DataFrame(data=data)
    predictor = createPredictor()
    assert predictor.df.equals(df)
    assert predictor.index == 'Timestamp'


def test_split_train_validation():
    df = pd.DataFrame(data=data)
    predictor = createPredictor()
    predictor.split_train_validation(testRecordsCount, trainRecordsCount)
    assert predictor.train.equals(df.head(3))
    assert predictor.test.equals(df.tail(2))    
    
def test_sample_train():
    df = pd.DataFrame(data=data)
    predictor = createPredictor()
    predictor.sample_train(3)
    assert predictor.train.equals(df.tail(3))

def test_set_date_as_index():
    predictor = createPredictor()
    predictor.train = predictor.df
    predictor.set_date_as_index()
    timeData = {
        'Timestamp': ['2011-12-31 07:52:00','2011-12-31 07:53:00','2011-12-31 07:54:00','2011-12-31 07:55:00','2011-12-31 07:56:00']
    }
    timestamps = pd.DataFrame(data=timeData)
    timestamps.Timestamp = pd.to_datetime(timestamps.Timestamp)
    assert predictor.train.Timestamp.equals(timestamps.Timestamp)
    
def test_clean_train():
    predictor = createPredictor()
    predictor.train = pd.DataFrame(data=missingData)
    predictor.clean_train()
    expectedDf = pd.DataFrame(data=backfilledData)
    expectedDf.Close = expectedDf.Close.astype(float)
    expectedDf.Open = expectedDf.Open.astype(float)
    assert predictor.train.equals(expectedDf)
    
    
def test_save_to_feather():
    df = pd.DataFrame(data=data)
    predictor = StockPredictor(df, index)
    predictor.train = df.tail(3)
#    predictor.save_to_feather('stockPredictor/')
#    indexDf = pd.DataFrame(data=[0,1,2])
#    assert predictor.train.index.equals(indexDf)
    
    
    
#    print('******')
#    print(filledDf.to_string())
#    print('******')
#    print(expectedDf.to_string())