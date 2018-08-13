#from fastai.structured import *
#from fastai.column_data import *
#np.set_printoptions(threshold=50, edgeitems=20)
#from ta import *
import pandas as pd

from stockPredictor import StockPredictor

config = {
    'testRecordsCount': 2,
    'trainRecordsCount': 3
}
data = {
    'Close': [1, 25, 6, 34, 6], 
    'Open': [3, 4, 78, 56, 7]
}    
trainData = {
    'Close': [1, 25, 6], 
    'Open': [3, 4, 78]
}
testData = {
    'Close': [1, 25, 6], 
    'Open': [3, 4, 78]
}

df = pd.DataFrame(data=data)
stock_predictor = StockPredictor(config,df)
    
def test_constructor():
    assert stock_predictor.config == config
    assert stock_predictor.df.equals(df)


def test_clean_data():
    trainDf = pd.DataFrame(data=trainData)
    testDf = pd.DataFrame(data=testData)
#    stock_predictor.split_train_validation()
#    assert stock_predictor.train == trainDf
#    assert stock_predictor.test == testDf