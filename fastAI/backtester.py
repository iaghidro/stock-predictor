
# coding: utf-8

# # BTC Predictor

# In[51]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[52]:


from fastai.structured import *
from fastai.column_data import *
np.set_printoptions(threshold=50, edgeitems=20)
from ta import *


# ## Stock Predictor Lib
# 

# In[53]:


def cleanData(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(method='bfill')
#     df = df.dropna()
#     df = df.replace(np.nan,df.mean())
    return df


# In[54]:


def calculateNetProfit(dataFrame, startAmount, fee):
    df = dataFrame
    df['buyAmount'] = 0
    df['sellAmount'] = 0
    totalBuys = 0
    totalSells = 0
    for index, row in df.iterrows():
        prevBuyAmount = df.buyAmount.get(index -1, np.nan)
        prevSellAmount = df.sellAmount.get(index -1, np.nan)
        predicted = row.action 
        if index == df.index[0]:
            df.loc[index,'buyAmount'] = startAmount
        elif predicted == 1 and prevBuyAmount > 0:
            # BUY
            sellAmount = prevBuyAmount/row.Close
            df.loc[index,'sellAmount'] = sellAmount - (sellAmount * fee)
#             df.loc[index,'sellAmount'] = prevBuyAmount/row.Close
            totalBuys +=1
        elif predicted == 1 and prevBuyAmount == 0:
            df.loc[index,'sellAmount'] = prevSellAmount
        elif predicted == 0 and prevSellAmount > 0:
            # SELL             
            buyAmount = prevSellAmount*row.Close
            df.loc[index,'buyAmount'] = buyAmount - (buyAmount * fee)
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
        'trades':totalBuys + totalSells
    }
    return df,result


# In[55]:


#  use conflateTimeFrame(df, '5T')
def conflateTimeFrame(df, timeFrame):
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

# In[56]:


pd.set_option('display.max_columns', 300)
pd.set_option('display.max_rows', 300)

lookahead = 10
percentIncrease = 1.002
# index='Timestamp'
index='time_period_start'
dep = 'action'
PATH='data/stock/'


# ## Create datasets

# In[57]:


table_names = [
#         'btc-bitstamp-2012-01-01_to_2018-01-08'
#     'BTC_2018_08-10_08-21'
#     'BTC_COINBASE_2018-07-25_09-06'
    'ETH_COINBASE_07-21_08-24'
]


# In[58]:


tables = [pd.read_csv(f'{PATH}{fname}.csv', low_memory=False) for fname in table_names]


# In[59]:


from IPython.display import HTML


# In[60]:


for t in tables: display(t.head())


# The following returns summarized aggregate information to each table accross each field.

# In[61]:


train= tables[0]


# In[62]:


len(train)


# In[63]:


# train = train.tail(700000)
train = train.tail(50000)
len(train)


# In[64]:


train.reset_index(inplace=True)
train.to_feather(f'{PATH}train')


# ## Data Cleaning

# In[65]:


train = pd.read_feather(f'{PATH}train')
train[index] = pd.to_datetime(train[index])
# train[index] = pd.to_datetime(train[index], unit='s')


# In[66]:


# edit columns

# train = pd.DataFrame({
#     'Timestamp':train[index],
#     'Close':train.Close
# })[['Timestamp','Close']]
# train.head(10)


train = pd.DataFrame({
    'Timestamp':train[index],
    'Close':train['price_close']
})[['Timestamp','Close']]
train.head(10)


# ## Conflate Time

# In[67]:


# train = train.set_index(pd.DatetimeIndex(train[index]))
# train = conflateTimeFrame(train, '5T')
# # fix this, should not have to extract Close
# train = train.Close 
# len(train)


# ## Set Target

# In[68]:


maxLookahead = train.Close.rolling(window=lookahead,min_periods=1).max() 
# maxLookahead = train.Close.iloc[::-1] \
#                             .rolling(window=lookahead,min_periods=1) \
#                             .max() \
#                             .iloc[::-1]
train['action'] =  maxLookahead > (percentIncrease * train['Close'])
train['max'] = maxLookahead
train.action = train.action.astype(int)

# target count by category
len(train[train.action==2]),len(train[train.action==1]),len(train[train.action==0])


# In[69]:


train.head(10)


# ## Validation

# In[70]:


valpred = pd.DataFrame({'Timestamp':train.Timestamp,'Close':train.Close, 'action':train.action})[['Timestamp','Close', 'action']]


# In[71]:


valpred.tail(10)


# In[72]:


newdf,result = calculateNetProfit(valpred, 100, 0)
result


# In[77]:


newdf.head(10)


# In[74]:


newdf.plot(x='Timestamp', y=['Close', 'buyAmount'], style='o',figsize=(10,5), grid=True)


# In[75]:


# temp = newdf.tail(90000)
# temp.head(300)

