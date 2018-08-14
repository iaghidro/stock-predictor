import numpy as np
import pandas as pd
#from fastai.structured import *
#from fastai.column_data import *
#np.set_printoptions(threshold=50, edgeitems=20)

#from ta import *

class StockPredictor:
    
    def __init__(self, df):
        self.df = df
        
        
    def split_train_validation(self, testRecordsCount, trainRecordsCount):
        self.test = self.df.tail(testRecordsCount)
        self.train = self.df.head(trainRecordsCount)
#        self.test.reset_index(inplace=True)
#        self.train.reset_index(inplace=True)
        print('StockPredictor::split_train_validation:: Train size: ' + str(len(self.train)) + ' Test size: ' + str(len(self.test)))
    
    def clean_data(self, df):
    #     df = df.dropna()
    #     df = df.replace(np.nan,df.mean())
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='bfill')
        return df
    