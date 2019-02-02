#!/usr/bin/env python
# coding: utf-8

# # Backtester

# In[47]:


import numpy as np
import pandas as pd
import os
import sys
module_path = os.path.abspath(os.path.join('lib/predictor'))
if module_path not in sys.path:
    sys.path.append(module_path)
from stockPredictor import StockPredictor 


# ## Config
# 

# In[48]:


np.set_printoptions(threshold=50, edgeitems=20)
pd.set_option('display.max_columns', 300)
pd.set_option('display.max_rows', 300)
from IPython.display import HTML


# In[49]:


index='Timestamp'
# index='time_period_start'
sample_size=60000
PATH='data/stock/'


# In[50]:


# # train.to_csv(f'{PATH}test.csv', sep=',', encoding='utf-8')

# p = StockPredictor(pd.DataFrame(), index)
# p.read_from_feather(PATH)
# p.train


# ## Create datasets

# In[51]:


table_names = [
#         'btc-bitstamp-2012-01-01_to_2018-01-08'
#     'BTC_COINBASE_2018-07-25_09-06'
#     'ETH_COINBASE_07-21_08-24'
#     'COINBASE_BCH_2018-06-15_09-01'
#     'COINBASE_BTC_2017-11-01_01-09'
    'bitstampUSD_1-min_data_2012-01-01_to_2018-06-27',
#     'coinbaseUSD_1-min_data_2014-12-01_to_2018-06-27'
#     'coinbaseUSD_1-min_data_2014-12-01_to_2018-11-11',
#     'bitstamp_07-09'
#     'btc_historical_parsed'
]


# In[52]:


tables = [pd.read_csv(f'{PATH}{fname}.csv', low_memory=False) for fname in table_names]


# In[53]:


# for t in tables: display(t.head())


# In[54]:


train= tables[0]
train


# In[55]:


p = StockPredictor(train, index)
p.sample_train(sample_size)
# p.train = p.train.head(100000)
p.normalize_train('Volume_(BTC)','Open','High','Low','Close', 'Weighted_Price')
p.train


# ## Change Point Analysis Setup

# In[56]:


# ! pip install ruptures
import ruptures as rpt
import matplotlib.pyplot as plt


# In[57]:


signal_df = pd.DataFrame({
    'Close':p.train.Close,
    'Volume':p.train.Volume
})[['Close','Volume']]
signal = signal_df.values


# #### Binary Segmentation

# In[58]:


model = "l2"  # "l1", "rbf", "linear", "normal", "ar"
algo = rpt.Binseg(model=model).fit(signal)
my_bkps = algo.predict(n_bkps=3)

# show results
rpt.show.display(signal, my_bkps, my_bkps, figsize=(10, 6))
plt.show()


# In[59]:


my_bkps


# #### Window Based

# In[60]:


# change point detection
model = "l2"  # "l1", "rbf", "linear", "normal", "ar"
algo = rpt.Window(width=40, model=model).fit(signal)
my_bkps = algo.predict(n_bkps=3)

# show results
rpt.show.display(signal, my_bkps, my_bkps, figsize=(10, 6))
plt.show()


# In[61]:


my_bkps


# In[ ]:




