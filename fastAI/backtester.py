
# coding: utf-8

# # BTC Predictor

# In[331]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[332]:


from fastai.structured import *
from fastai.column_data import *
np.set_printoptions(threshold=50, edgeitems=20)
from ta import *


# ## Stock Predictor Lib
# 

# In[333]:


import json as js
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
        self.train[self.index] = pd.to_datetime(self.train[self.index])

    def set_date_as_index_unix(self):
        self.train[self.index] = pd.to_datetime(self.train[self.index], unit='s')
        
    def split_train_validation(self, testRecordsCount, trainRecordsCount):
        self.test = self.df.tail(testRecordsCount)
        self.train = self.df.head(trainRecordsCount)
#        self.test.reset_index(inplace=True)
#        self.train.reset_index(inplace=True)
        print('StockPredictor::split_train_validation:: Train size: ' + str(len(self.train)) + ' Test size: ' + str(len(self.test)))    
        
    def normalize_train(self, volume, open, high, low, close):
        self.train = pd.DataFrame({
            'Timestamp':self.train[self.index],
            'Volume':self.train[volume],
            'Open':self.train[open],
            'High':self.train[high],
            'Low':self.train[low],
            'Close':self.train[close]
        })[['Timestamp','Volume','Open','High','Low','Close']]

    def clean_train(self):
    #     df = df.dropna()
    #     df = df.replace(np.nan,df.mean())
        self.train = self.train.replace([np.inf, -np.inf], np.nan)
        self.train = self.train.fillna(method='bfill')
                        
    # ///////////////////////////////
    # //// FEATURE ENGINEERING //////
    # ///////////////////////////////
    
    def add_ta(self):
        self.train = add_all_ta_features(self.train, "Open", "High", "Low", "Close", "Volume", fillna=True)
        
    """ Set the target (dependent variable) by looking ahead in a certain time window and percent increase
        to determine if the action should be a BUY or a SELL. BUY is true/1 SELL is false/0""" 
    def set_target(self, lookahead, percentIncrease):
#        ,win_type='boxcar'
        max_in_lookahead_timeframe = self.train.Close                                                 .iloc[::-1]                                                 .rolling(window=lookahead,min_periods=1)                                                 .max()                                                 .iloc[::-1]
        self.train['action'] = max_in_lookahead_timeframe > (percentIncrease * self.train.Close)
#        self.train['max'] =max_in_lookahead_timeframe
        self.train.action = self.train.action.astype(int)

    def set_target_historical(self, lookahead, percentIncrease):
        max_in_lookback_timeframe = self.train.Close.rolling(window=lookahead,min_periods=1).max()
        self.train['action'] = max_in_lookback_timeframe > (percentIncrease * self.train.Close)
        self.train.action = self.train.action.astype(int)
        buy_count = str(len(self.train[self.train.action==1]))
        sell_count = str(len(self.train[self.train.action==0]))
        print('StockPredictor::set_target_historical:: Buy count: ' + buy_count + ' Sell count: ' + sell_count)
        
    # ///////////////////////////////
    # ///////// EVALUATION //////////
    # ///////////////////////////////
    
    def calculate_accuracy(self, df):
        successful_predictions = df.loc[df.action == df.predicted]
        total_accuracy = len(successful_predictions)/len(df)
        total_buy_actions = df.loc[df.action == 1]
        total_sell_actions = df.loc[df.action == 0]
        successful_buy_predictions = successful_predictions.loc[successful_predictions.action == 1]
        successful_sell_predictions = successful_predictions.loc[successful_predictions.action == 0]
        buy_accuracy = len(successful_buy_predictions)/len(total_buy_actions)
        sell_accuracy = len(successful_sell_predictions)/len(total_sell_actions)
        f1Score = (buy_accuracy + sell_accuracy)/2;
        result = {
            'F1Score': round(f1Score,3),
            'totalAccuracy': round(total_accuracy,3),
            'buyAccuracy': round(buy_accuracy,3),
            'sellAccuracy': round(sell_accuracy,3),
            'totalBuyActions': len(total_buy_actions),
            'successfulBuyPredictions': len(successful_buy_predictions)
        }
        return result
            
    def calculate_net_profit(self, inputDf, startAmount):
        df = inputDf
        df['buyAmount'] = 0
        df['sellAmount'] = 0
        totalBuys = 0
        totalSells = 0
        for index, row in df.iterrows():
            prevBuyAmount = df.buyAmount.get(index -1, np.nan)
            prevSellAmount = df.sellAmount.get(index -1, np.nan)
    #         prevPredicted = df.predicted.get(index -1, np.nan)
            predicted = row.predicted
            if index == df.index[0]:
                df.loc[index,'buyAmount'] = startAmount
            elif predicted == 1 and prevBuyAmount > 0:
                # BUY
                df.loc[index,'sellAmount'] = prevBuyAmount/row.Close
                totalBuys +=1
            elif predicted == 1 and prevBuyAmount == 0:
                df.loc[index,'sellAmount'] = prevSellAmount
            elif predicted == 0 and prevSellAmount > 0:
                # SELL             
                df.loc[index,'buyAmount'] = prevSellAmount*row.Close
                totalSells +=1
            elif predicted == 0 and prevSellAmount == 0:
                df.loc[index,'buyAmount'] = prevBuyAmount
            else:
                # HOLD (not holding currently)
                df.loc[index,'buyAmount'] = prevBuyAmount
                df.loc[index,'sellAmount'] = prevSellAmount
                
        startClose = df.Close.iloc[0]
        endClose = df.Close.iloc[-1]
        endBuyAmount = df.buyAmount.iloc[-1]
        endSellAmount = df.sellAmount.iloc[-1]
        endAmount = endBuyAmount if (endBuyAmount > 0) else (endSellAmount * endClose)
        buyAndHoldPercentIncrease = ((endClose - startClose)/startClose) * 100
        percentIncrease = ((endAmount - startAmount)/startAmount) * 100
        percentDifference = percentIncrease - buyAndHoldPercentIncrease

        result = {
            'startClose': startClose,
            'endClose': endClose,
            'startAmount': startAmount,
            'endAmount': endAmount,
            'buyAndHoldPercentIncrease':round(buyAndHoldPercentIncrease,3),
            'percentIncrease':round(percentIncrease,3),
            'percentDifference':round(percentDifference,3),
            'totalTrades':totalBuys + totalSells
        }
        self.net_profit_df = df
        self.result = result
        print(js.dumps(result, sort_keys=False,indent=4, separators=(',', ': ')))

    # ///////////////////////////////
    # /////////// UTIL //////////////
    # ///////////////////////////////
    
    def save_to_feather(self):
        self.train.reset_index(inplace=True)
        self.train.to_feather(f'{PATH}train')
    
    def read_from_feather(self):
        self.train = pd.read_feather(f'{PATH}train')
        # train.drop(self.index,1,inplace=True)
    
    """ usage conflateTimeFrame(df, '5T') """
    def conflate_time_frame(self, df, timeFrame):
        ohlc_dict = {                                                                                                             
            'Open':'first',                                                                                                    
            'High':'max',                                                                                                       
            'High':'max',                                                                                                       
            'Low':'min',                                                                                                        
            'Close': 'last',                                                                                                    
            'Volume': 'sum'
        }
        return df.resample(timeFrame).agg(ohlc_dict)


# ## Config
# 

# In[334]:


pd.set_option('display.max_columns', 300)
pd.set_option('display.max_rows', 300)

lookahead = 10
percentIncrease = 1.002
# index='Timestamp'
index='time_period_start'
dep = 'action'
PATH='data/stock/'


# ## Create datasets

# In[335]:


table_names = [
#         'btc-bitstamp-2012-01-01_to_2018-01-08'
#     'BTC_2018_08-10_08-21'
#     'BTC_COINBASE_2018-07-25_09-06'
    'ETH_COINBASE_07-21_08-24'
]


# In[336]:


tables = [pd.read_csv(f'{PATH}{fname}.csv', low_memory=False) for fname in table_names]


# In[337]:


from IPython.display import HTML


# In[338]:


for t in tables: display(t.head())


# The following returns summarized aggregate information to each table accross each field.

# In[339]:


train= tables[0]


# In[340]:


len(train)


# In[341]:


p = StockPredictor(train, index)
p.sample_train(50000)


# In[342]:


p.save_to_feather()


# ## Data Cleaning

# In[343]:


p.read_from_feather()
p.set_date_as_index()
# p.set_date_as_index_unix()


# In[344]:


p.normalize_train('volume_traded','price_open','price_high','price_low','price_close',)
p.train.head(10)


# ## Conflate Time

# In[345]:


# train = train.set_index(pd.DatetimeIndex(train[index]))
# train = conflateTimeFrame(train, '5T')
# # fix this, should not have to extract Close
# train = train.Close 
# len(train)


# ## Set Target

# In[ ]:


p.set_target_historical(lookahead, percentIncrease)
# p.train


# In[ ]:


p.train.head(10)


# ## Validation

# In[ ]:


valpred = pd.DataFrame({
    'Timestamp':p.train.Timestamp,
    'Close':p.train.Close,
    'action':p.train.action,
    'predicted':p.train.action
})
[['Timestamp','Close', 'action']]


# In[ ]:


valpred.tail(10)


# In[ ]:


p.calculate_net_profit(valpred, 100)


# In[ ]:


p.net_profit_df.head(10)


# In[ ]:


p.net_profit_df.plot(x='Timestamp', y=['Close', 'buyAmount'], style='o',figsize=(10,5), grid=True)

