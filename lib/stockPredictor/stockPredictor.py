#from fastai.structured import *
#from fastai.column_data import *
#np.set_printoptions(threshold=50, edgeitems=20)

#from ta import *

class StockPredictor:
    
    def __init__(self, config, df):
        self.config = config
        self.df = df
        
        
    def split_train_validation(self):
        self.test = self.df.tail(self.config.testRecordsCount)
        self.test.reset_index(inplace=True)
        self.train = self.df.head(self.config.trainRecordsCount)
        self.train.reset_index(inplace=True)
        print('Train size: ' + len(self.train) + ' Test size: ' + len(self.test))
    
    #     df = df.dropna()
    #     df = df.replace(np.nan,df.mean())
    def clean_data(self, df):
        df = df.replace([np.inf, -np.inf], np.nan)
        df.fillna(method='bfill')
        return df