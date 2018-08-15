#from fastai.structured import *
#from fastai.column_data import *
#np.set_printoptions(threshold=50, edgeitems=20)

import pandas as pd
import numpy as np

from stockPredictor import StockPredictor

data = {
    'Close': [1, 25, 6, 34, 6], 
    'Open': [3, 4, 78, 56, 7]
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

def createPredictor():
    df = pd.DataFrame(data=data)
    return StockPredictor(df)
    
def test_constructor():
    df = pd.DataFrame(data=data)
    predictor = StockPredictor(df)
    assert predictor.df.equals(df)


def test_split_train_validation():
    df = pd.DataFrame(data=data)
    predictor = createPredictor()
    predictor.split_train_validation(testRecordsCount, trainRecordsCount)
    assert predictor.train.equals(df.head(3))
    assert predictor.test.equals(df.tail(2))
    
    
#def test_add_TA():
#    predictor.add_TA()
    
def test_sample_train():
    df = pd.DataFrame(data=data)
    predictor = createPredictor()
    predictor.sample_train(3)
    assert predictor.train.equals(df.tail(3))
    
def test_clean_train():
    predictor = createPredictor()
    predictor.train = pd.DataFrame(data=missingData)
    predictor.clean_train()
    expectedDf = pd.DataFrame(data=backfilledData)
    expectedDf.Close = expectedDf.Close.astype(float)
    expectedDf.Open = expectedDf.Open.astype(float)
    assert predictor.train.equals(expectedDf)
    
    
    
#    print('******')
#    print(filledDf.to_string())
#    print('******')
#    print(expectedDf.to_string())