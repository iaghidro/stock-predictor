#!/usr/bin/env python
# coding: utf-8

# # BTC Predictor

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


from fastai.structured import *
from fastai.column_data import *
np.set_printoptions(threshold=50, edgeitems=20)
from ta import *
from IPython.display import HTML
from IPython.core.display import display

import os
import sys
module_path = os.path.abspath(os.path.join('lib/predictor'))
if module_path not in sys.path:
    sys.path.append(module_path)
from stockPredictor import StockPredictor 


# ## Config
# 

# In[3]:


pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

index='Timestamp'
# index='time_period_start'
lookahead = 20
percentIncrease = 1.001
recordsCount = 50000
test_ratio  = 0.95
train_ratio = 0.95
lr = 1e-4
dropout = 0.04
modelName = 'btcBinaryClassificationModel'
dep = 'action'
PATH='data/stock/'


# In[4]:


cat_vars = ['TimestampYear', 'TimestampMonth', 'TimestampWeek', 'TimestampDay', 'hour','minute', 'TimestampDayofweek',
'TimestampDayofyear','TimestampIs_month_end', 'TimestampIs_month_start', 'TimestampIs_quarter_end',
'TimestampIs_quarter_start','TimestampIs_year_end', 'TimestampIs_year_start']

contin_vars = ['Open', 'Close','High', 'Low', 'Volume', 'TimestampElapsed',
'volume_adi','volume_obv','volume_obvm','volume_cmf','volume_fi','volume_em','volume_vpt','volume_nvi',
'volatility_atr','volatility_bbh','volatility_bbl','volatility_bbm','volatility_bbhi','volatility_bbli',
'volatility_kcc','volatility_kch','volatility_kcl','volatility_kchi','volatility_kcli','volatility_dch',
'volatility_dcl','volatility_dchi','volatility_dcli','trend_macd','trend_macd_signal','trend_macd_diff',
'trend_ema_fast','trend_ema_slow','trend_adx','trend_vortex_ind_pos',
'trend_vortex_ind_neg','trend_vortex_diff','trend_trix','trend_mass_index','trend_cci','trend_dpo','trend_kst',
'trend_kst_sig','trend_kst_diff','trend_ichimoku_a','trend_ichimoku_b'
,'trend_aroon_up','trend_aroon_down','trend_aroon_ind','momentum_rsi','momentum_mfi','momentum_tsi'
,'momentum_uo','momentum_stoch','momentum_stoch_signal','momentum_wr','momentum_ao','others_dr','others_dlr','others_cr'
,'sma5','sma15','sma30','sma60','sma90'
,'1Open','1High','1Low','1Close','1Volume','2Open','2High','2Low','2Close','2Volume'
,'3Open','3High','3Low','3Close','3Volume','4Open','4High','4Low','4Close','4Volume'
,'5Open','5High','5Low','5Close','5Volume','6Open','6High','6Low','6Close','6Volume'
,'7Open','7High','7Low','7Close','7Volume','8Open','8High','8Low','8Close','8Volume','9Open','9High','9Low'
,'9Close','9Volume','10Open','10High','10Low','10Close','10Volume','11Open','11High','11Low','11Close','11Volume'
,'12Open','12High','12Low','12Close','12Volume','13Open','13High','13Low','13Close','13Volume'
,'14Open','14High','14Low','14Close','14Volume','15Open','15High','15Low','15Close','15Volume'
,'16Open','16High','16Low','16Close','16Volume','17Open','17High','17Low','17Close','17Volume'
,'18Open','18High','18Low','18Close','18Volume','19Open','19High','19Low','19Close','19Volume'
,'20Open','20High','20Low','20Close','20Volume','21Open','21High','21Low','21Close','21Volume'
,'22Open','22High','22Low','22Close','22Volume','23Open','23High','23Low','23Close','23Volume'
,'24Open','24High','24Low','24Close','24Volume','25Open','25High','25Low','25Close','25Volume'
,'26Open','26High','26Low','26Close','26Volume','27Open','27High','27Low','27Close','27Volume'              
]

len(cat_vars),len(contin_vars)


# ## Create datasets

# In[5]:


table_names = [
#     'coinbaseUSD_1-min_data_2014-12-01_to_2018-06-27',
    'coinbaseUSD_1-min_data_2014-12-01_to_2018-11-11',
#     'bitstampUSD_1-min_data_2012-01-01_to_2018-06-27',
#     'btc-bitstamp-2012-01-01_to_2018-01-08'
#         'BTC_COINBASE_2018-07-25_09-06'
#         'ETH_COINBASE_07-21_08- 24'
]


# In[6]:


tables = [pd.read_csv(f'{PATH}{fname}.csv', low_memory=False) for fname in table_names]


# In[7]:


for t in tables: display(t.head())


# In[8]:


train= tables[0]


# In[9]:


# train = train.head(1700000)
p = StockPredictor(train, index)
p.sample_train(recordsCount)


# ## Data Cleaning

# In[10]:


p.set_date_as_index_unix()
p.normalize_train('Volume_(BTC)','Open','High','Low','Close', 'Weighted_Price')
# p.set_date_as_index()
# p.normalize_train('volume_traded','price_open','price_high','price_low','price_close', 'price_close')
p.train.head()


# ## Feature Engineering

# In[11]:


# add technical analysis
p.add_ta()
p.clean_train()


# In[12]:


p.set_target_hold('Close',lookahead, percentIncrease, 1)


# In[13]:


p.add_date_values()
p.trim_ends(100,100)


# In[14]:


# p.train.to_csv(f'{PATH}btc_historical_parsed.csv', sep=',', encoding='utf-8')
# p.save_to_feather(PATH)
# p.read_from_feather(PATH)
p.train


# ## Split validation and test sets

# In[15]:


p.split_test_train(test_ratio)


# In[16]:


# p.train.head()


# In[17]:


p.train.tail(50).T.head(100)


# ## Create features

# In[18]:


p.train = p.apply_variable_types(p.train, cat_vars, contin_vars, dep)
p.test = p.apply_variable_types(p.test, cat_vars, contin_vars, dep)
apply_cats(p.test, p.train)


# In[19]:


df, y, nas, mapper = proc_df(p.train, dep, do_scale=True)


# In[20]:


df_test, _, nas, mapper = proc_df(p.test, dep, do_scale=True, mapper=mapper, na_dict=nas)
nas={}
# p.train.head(30).T.head(70)


# In[21]:


df.head(2)


# In[22]:


df_test.head(2)


# Rake the last x% of rows as our validation set.

# In[23]:


train_size = p.get_train_size(train_ratio)
val_idx = p.get_validation_indexes(train_size, df)


# ## DL

# We can create a ModelData object directly from our data frame. Is_Reg is set to False to turn this into a classification problem (from a regression).

# In[24]:


# y = y.reshape(len(y),1)
md = ColumnarModelData.from_data_frame(PATH, val_idx, df,y.astype('int'), cat_flds=cat_vars, bs=128,
                                      is_reg=False,is_multi=False,test_df=df_test)


# Some categorical variables have a lot more levels than others.

# In[25]:


cat_sz = [(c, len(p.train[c].cat.categories)+1) for c in cat_vars]
emb_szs = [(c, min(50, (c+1)//2)) for _,c in cat_sz]
cont_size = len(df.columns)-len(cat_vars)


# In[26]:


activations=[500,100]
dropout_later_layers= [0.01,0.1] 
m = md.get_learner(emb_szs, cont_size,dropout, 3, activations,dropout_later_layers,None,True)

from torch.nn import functional as F
m.crit = F.cross_entropy


# In[27]:


# def accuracy(preds, targs, thresh=0.5):
#     return ((preds>thresh).float()==targs).float().mean()

def accuracy(preds, targs):
    preds = torch.max(preds, dim=1)[1]
    return (preds==targs).float().mean()


# In[28]:


m.lr_find()
m.sched.plot(10)


# In[29]:


m.fit(lr, 2, metrics=[accuracy])


# In[30]:


m.fit(lr, 5, cycle_len=1, metrics=[accuracy])


# In[31]:


m.fit(lr, 3, cycle_len=4, cycle_mult=2 , metrics=[accuracy])


# In[32]:


m.save(modelName)


# In[33]:


m.load(modelName)


# ## Validation

# In[34]:


(x,yl)=m.predict_with_targs()
# x


# In[35]:


val = p.train.iloc[val_idx]
val[[dep]]
valpred = pd.DataFrame({
    'Close':val.Close,
    'index':val.index,
    'action':val.action,
    'predicted': np.argmax(x,axis=1),
})[['Close','index', 'action','predicted']]
# valpred


sell_count = str(len(valpred[valpred.predicted == 0]))
hold_count = str(len(valpred[valpred.predicted == 1]))
buy_count = str(len(valpred[valpred.predicted == 2]))
print('Buy count: ' + buy_count + ' Sell count: ' + sell_count + ' Hold count: ' + hold_count)
valpred


# Calculate the percent accuracy on the validation set

# In[36]:


p.calculate_accuracy_hold(valpred)


# In[37]:


p.calculate_net_profit_hold(valpred, 15000, 0)
p.result


# In[38]:


p.plot_profit(p.net_profit_df)


# In[39]:


p.net_profit_df


# ## Test

# In[40]:


np.argmax(m.predict(True), axis =1)


# In[41]:


testPred = pd.DataFrame({
    'index':p.test.index,
    'Close':p.test.Close,
    'action':p.test.action, 
    'predicted':np.argmax(m.predict(True), axis =1)
})[['index','Close','action', 'predicted']]
testPred.head(10)


# In[42]:


p.calculate_accuracy_hold(testPred)


# In[43]:


p.calculate_net_profit_hold(testPred, 15000, 0)
p.result


# In[44]:


p.net_profit_df


# In[45]:


p.plot_profit(p.net_profit_df)


# In[46]:


p.confusion_matrix(testPred.action, testPred.predicted, ['0', '1', '2'], [0, 1, 2])


# ## RF

# In[47]:


from sklearn.ensemble import RandomForestClassifier
mrf = RandomForestClassifier(n_estimators=150, max_features=0.99, min_samples_leaf=2,
                          n_jobs=-1, oob_score=True)
mrf.fit(df.values, y);
mrf


# In[48]:


mrf.score(df_test.values,p.test[dep])


# In[49]:


preds = mrf.predict(df_test.values)
testPred.predicted = preds


# In[50]:


p.confusion_matrix(testPred.action, testPred.predicted, ['0', '1', '2'], [0, 1, 2])


# In[51]:


p.calculate_accuracy_hold(testPred)


# In[52]:


p.calculate_net_profit_hold(testPred, 15000, 0)
p.result


# ## AdaBoost

# In[53]:


from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(random_state=1, learning_rate=0.5)
ada.fit(df.values, y)
ada


# In[54]:


ada.score(df_test.values,p.test[dep])


# In[55]:


preds = ada.predict(df_test.values)
testPred.predicted = preds


# In[56]:


p.confusion_matrix(testPred.action, preds, ['0', '1', '2'], [0, 1, 2])


# In[57]:


p.calculate_accuracy_hold(testPred)


# In[58]:


p.calculate_net_profit_hold(testPred, 15000, 0)
p.result


# ## Playground

# In[59]:


# list(p.train.columns.values)
m.crit


# In[60]:


# val_idx
# y

