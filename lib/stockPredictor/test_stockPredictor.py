#from fastai.structured import *
#from fastai.column_data import *
#np.set_printoptions(threshold=50, edgeitems=20)

import numpy as np
import pandas as pd
from stockPredictor import StockPredictor

trainData = {
    'Close': [1, 1.25, 1.5, 2, 2.25], 
    'Open': [1.1, 1.2, 1.4, 1.8, 2],
    'Timestamp': [1325317920, 1325317980, 1325318040, 1325318100, 1325318160]
} 
baseData = {
    'Close': [1, 1.25, 1.5, 2, 2.25], 
    'Open': [1.1, 1.2, 1.4, 1.8, 2],
    'Timestamp': [1325317920, 1325317980, 1325318040, 1325318100, 1325318160],
    'action':    [1, 1, 0, 1, 0],
    'predicted': [0, 1, 0, 1, 1],
} 
missingData = {
    'Close': [1, np.nan, 6, 34, 6], 
    'Open': [np.nan, -np.inf, 78, 56, 7]
}  
backfilledData = {
    'Close': [1, 6, 6, 34, 6], 
    'Open': [78, 78, 78, 56, 7]
}    
testRecordsCount = 2
trainRecordsCount = 3
index = 'Timestamp'

def create_predictor(inputData):
    df = pd.DataFrame(data=inputData)
    return StockPredictor(df, index)
    
def test_constructor():
    df = pd.DataFrame(data=trainData)
    predictor = create_predictor(df)
    assert predictor.df.equals(df)
    assert predictor.index == 'Timestamp'

# ///////////////////////////////
# /////// DATA CLEANING /////////
# ///////////////////////////////

def test_split_train_validation():
    df = pd.DataFrame(data=trainData)
    predictor = create_predictor(df)
    predictor.split_train_validation(testRecordsCount, trainRecordsCount)
    assert predictor.train.equals(df.head(3))
    assert predictor.test.equals(df.tail(2))    
    
def test_sample_train():
    df = pd.DataFrame(data=trainData)
    predictor = create_predictor(df)
    predictor.sample_train(3)
    assert predictor.train.equals(df.tail(3))

def test_set_date_as_index():
    predictor = create_predictor(pd.DataFrame(data=trainData))
    predictor.train = predictor.df
    predictor.set_date_as_index()
    timeData = {
        'Timestamp': ['2011-12-31 07:52:00', '2011-12-31 07:53:00', '2011-12-31 07:54:00', '2011-12-31 07:55:00', '2011-12-31 07:56:00']
    }
    timestamps = pd.DataFrame(data=timeData)
    timestamps.Timestamp = pd.to_datetime(timestamps.Timestamp)
    assert predictor.train.Timestamp.equals(timestamps.Timestamp)
    
def test_clean_train():
    predictor = create_predictor(trainData)
    predictor.train = pd.DataFrame(data=missingData)
    predictor.clean_train()
    expectedDf = pd.DataFrame(data=backfilledData)
    expectedDf.Close = expectedDf.Close.astype(float)
    expectedDf.Open = expectedDf.Open.astype(float)
    assert predictor.train.equals(expectedDf)
                    
# ///////////////////////////////
# //// FEATURE ENGINEERING //////
# ///////////////////////////////

def test_set_target():
    df = pd.DataFrame(data=trainData)
    predictor = StockPredictor(df, index)
    predictor.train = df.tail(3)
    
# ///////////////////////////////
# ///////// EVALUATION //////////
# ///////////////////////////////

def test_calculate_accuracy():
    df = pd.DataFrame(data=baseData)
    predictor = create_predictor(baseData)
    result = predictor.calculate_accuracy(df)
    expected = {
        'F1Score': .583,
        'totalAccuracy': .6,
        'buyAccuracy': .667,
        'sellAccuracy': .5,
        'totalBuyActions': 3,
        'successfulBuyPredictions': 2
    }
    assert result == expected
    
def test_calculate_net_profit():
    df = pd.DataFrame(data=baseData)
    predictor = create_predictor(baseData)
#    df.reset_index(inplace=True)
    result = predictor.calculate_net_profit(df, 100)
    expected = {
        'startClose': 1,
        'endClose': 2.25,
        'startAmount': 100,
        'endAmount': 135,
        'buyAndHoldPercentIncrease': 125,
        'percentIncrease':35,
        'percentDifference': -90,
        'totalTrades': 3
    }
    assert result == expected
    
# ///////////////////////////////
# /////////// UTIL //////////////
# ///////////////////////////////
    
def test_save_to_feather():
    df = pd.DataFrame(data=baseData)
    predictor = StockPredictor(df, index)
    predictor.train = df.tail(3)
#    predictor.save_to_feather('stockPredictor/')
#    indexDf = pd.DataFrame(data=[0,1,2])
#    assert predictor.train.index.equals(indexDf)
    
    
    
#    print('******')
#    print(filledDf.to_string())
#    print('******')
#    print(expectedDf.to_string())