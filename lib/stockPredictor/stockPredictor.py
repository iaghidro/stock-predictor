import numpy as np
import pandas as pd
#from fastai.structured import *
#from fastai.column_data import *
#np.set_printoptions(threshold=50, edgeitems=20)
from ta import *

class StockPredictor:
    
    def __init__(self, df, index):
        self.df = df
        self.index = index        
                
    # ///////////////////////////////
    # /////// DATA CLEANING /////////
    # ///////////////////////////////
    
    def sample_train(self, sampleSize):
        self.train = self.df.tail(sampleSize)
        print('StockPredictor::sample_train:: Train size: ' + str(len(self.train)) + ' Original size: ' + str(len(self.df)))
        
    def set_date_as_index(self):
        self.train[self.index] = pd.to_datetime(self.train[self.index], unit='s')
        
    def split_train_validation(self, testRecordsCount, trainRecordsCount):
        self.test = self.df.tail(testRecordsCount)
        self.train = self.df.head(trainRecordsCount)
#        self.test.reset_index(inplace=True)
#        self.train.reset_index(inplace=True)
        print('StockPredictor::split_train_validation:: Train size: ' + str(len(self.train)) + ' Test size: ' + str(len(self.test)))    

    def add_ta(self):
        self.train = add_all_ta_features(self.train, "Open", "High", "Low", "Close", "Volume", fillna=True)
        
    def clean_train(self):
    #     df = df.dropna()
    #     df = df.replace(np.nan,df.mean())
        self.train = self.train.replace([np.inf, -np.inf], np.nan)
        self.train = self.train.fillna(method='bfill')
        

    # ///////////////////////////////
    # /////////// UTIL //////////////
    # ///////////////////////////////
    
    def save_to_feather(self):
        self.train.reset_index(inplace=True)
        self.train.to_feather(f'{PATH}train')
    
    def read_from_feather(self):
        self.train = pd.read_feather(f'{PATH}train')
        train.drop(self.index,1,inplace=True)