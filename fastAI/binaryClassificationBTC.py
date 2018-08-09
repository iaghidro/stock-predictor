
# coding: utf-8

# # BTC Predictor

# In[459]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[460]:


from fastai.structured import *
from fastai.column_data import *
np.set_printoptions(threshold=50, edgeitems=20)
from ta import *

PATH='data/stock/'


# ## Stock Predictor Lib
# 

# In[461]:


def cleanData(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    df.fillna(method='bfill')
#     df = df.dropna()
#     df = df.replace(np.nan,df.mean())
    return df


# In[462]:


def calculateAccuracy(df):
    successfulPredictions = df.loc[df.action == df.predicted]
    # total accuracy does not provide an accurate represantation
    # totalAccuracy = len(successfulPredictions)/len(df)
    totalBuyActions = df.loc[df.action == 1]
    totalSellActions = df.loc[df.action == 0]
    successfulBuyPredictions = successfulPredictions.loc[successfulPredictions.action == 1]
    successfulSellPredictions = successfulPredictions.loc[successfulPredictions.action == 0]
    buyAccuracy = len(successfulBuyPredictions)/len(totalBuyActions)
    sellAccuracy = len(successfulSellPredictions)/len(totalSellActions)
    result = {
        'F1Score': (buyAccuracy + sellAccuracy )/2,
        'buyAccuracy': buyAccuracy,
        'sellAccuracy': sellAccuracy,
        'totalBuyActions': len(totalBuyActions),
        'successfulBuyPredictions': len(successfulBuyPredictions)
    }
    return result
            
def calculateNetProfit(dataFrame, startAmount):
    df = dataFrame
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
        'totalBuys':totalBuys,
        'totalSells':totalSells
    }
    return df,result


# In[463]:


#  use conflateTimeFrame(df, '5T')
def conflateTimeFrame(df, timeFrame):
    ohlc_dict = {                                                                                                             
        'Open':'first',                                                                                                    
        'High':'max',                                                                                                       
        'Low':'min',                                                                                                        
        'Close': 'last',                                                                                                    
        'Volume': 'sum'
    }
    return df.resample(timeFrame).agg(ohlc_dict)


# ## Config
# 

# In[464]:


pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

lookahead = 25
percentIncrease = 1.003
recordsCount = 110000
testRecordsCount = 10000
trainRecordsCount = 100000
trainRatio = 0.9
lr = 1e-3
dropout = 0.09
modelName = 'btcBinaryClassificationModel'
index='Timestamp'
dep = 'action'


# ## Create datasets

# In[465]:


table_names = ['btc-bitstamp-2012-01-01_to_2018-01-08']


# In[466]:


tables = [pd.read_csv(f'{PATH}{fname}.csv', low_memory=False) for fname in table_names]


# In[467]:


from IPython.display import HTML


# In[468]:


for t in tables: display(t.head())


# The following returns summarized aggregate information to each table accross each field.

# In[469]:


# for t in tables: display(DataFrameSummary(t).summary())


# In[470]:


train= tables[0]


# In[471]:


len(train)


# In[472]:


# trim to x records for now
# TODO: remove this
# train = train.tail(1000000)
train = train.tail(recordsCount)
len(train)


# In[473]:


train.reset_index(inplace=True)
train.to_feather(f'{PATH}train')


# ## Data Cleaning

# In[474]:


train = pd.read_feather(f'{PATH}train')


# In[475]:


#convert to date objects
train[index] = pd.to_datetime(train[index], unit='s')
train.head()


# SET DEPENDENT VARIABLE ACTION

# In[476]:


# edit columns
train["Volume"] = train["Volume_(BTC)"]
train.drop('Volume_(BTC)',1,inplace=True)
train["VolumeCurrency"] = train["Volume_(Currency)"]
train.drop('Volume_(Currency)',1,inplace=True)

# delete unused columns
train.drop('VolumeCurrency',1,inplace=True)
train.drop('Weighted_Price',1,inplace=True)

train.head()


# In[477]:


# train = conflateTimeFrame(train, '5T')
# train[index] = train.index
# train.head()


# In[478]:


len(train)


# ## Feature Engineering

# In[479]:


# add technical analysis
train = add_all_ta_features(train, "Open", "High", "Low", "Close", "Volume", fillna=True)
train = cleanData(train)
len(train)


# In[480]:


# train['action'] = 0;
# train.loc[train['Close'].rolling(window=lookahead).max() > train['Close'], 'action'] = 1
# train.loc[train['Close'].rolling(window=lookahead).max() > percentIncrease * train['Close'], 'action'] = 2

train['action'] =  train['Close'].rolling(window=lookahead).max() > percentIncrease * train['Close']
train.action = train.action.astype(int)

# target count by category
len(train[train.action==2]),len(train[train.action==1]),len(train[train.action==0])


# Time modifications

# In[481]:


# add all date time values
add_datepart(train, index, drop=False)
train['hour'] = train[index].dt.hour;
train['minute'] = train[index].dt.minute;
len(train)


# ## Split validation and test sets

# In[482]:


# # todo: make this into a percentage instead of hardcoding the test set 
# # todo: create function 
test = train.tail(testRecordsCount)
test.reset_index(inplace=True)
train = train.head(trainRecordsCount)
train.reset_index(inplace=True)
len(train),len(test)


# In[483]:


train.to_feather(f'{PATH}train')
test.to_feather(f'{PATH}test')


# ## Create features

# In[484]:


train = pd.read_feather(f'{PATH}train')
test = pd.read_feather(f'{PATH}test')


# In[485]:


train.tail(50).T.head(100)


# In[486]:


# display(DataFrameSummary(train).summary())
# break break break now


# Now that we've engineered all our features, we need to convert to input compatible with a neural network.
# 
# This includes converting categorical variables into contiguous integers or one-hot encodings, normalizing continuous features to standard normal, etc...

# In[487]:


train.head()


# Identify categorical vs continuous variables

# In[488]:


cat_vars = ['TimestampYear', 'TimestampMonth', 'TimestampWeek', 'TimestampDay', 'hour','minute', 'TimestampDayofweek',
'TimestampDayofyear','TimestampIs_month_end', 'TimestampIs_month_start', 'TimestampIs_quarter_end',
'TimestampIs_quarter_start','TimestampIs_year_end', 'TimestampIs_year_start']

# techincal_indicators = ['volume_adi','volume_obv','volume_obvm','volume_cmf','volume_nvi','volatility_bbh',
# 'volatility_bbl','volatility_atr','volatility_bbm','trend_mass_index','trend_macd','trend_macd_signal',
# 'trend_kst','trend_kst_sig','trend_kst_diff','trend_macd_diff','trend_ema_fast','trend_ema_slow','trend_adx',
# 'trend_adx_pos','trend_adx_neg','trend_adx_ind','trend_ichimoku_a','trend_ichimoku_b','momentum_rsi','momentum_mfi',
# 'momentum_tsi','momentum_uo','momentum_stoch','momentum_stoch_signal','momentum_wr','momentum_ao']

contin_vars = ['Open', 'Close','High', 'Low', 'Volume', 'TimestampElapsed',
'volume_adi','volume_obv','volume_obvm','volume_cmf','volume_fi','volume_em','volume_vpt','volume_nvi',
'volatility_atr','volatility_bbh','volatility_bbl','volatility_bbm','volatility_bbhi','volatility_bbli',
'volatility_kcc','volatility_kch','volatility_kcl','volatility_kchi','volatility_kcli','volatility_dch',
'volatility_dcl','volatility_dchi','volatility_dcli','trend_macd','trend_macd_signal','trend_macd_diff',
'trend_ema_fast','trend_ema_slow','trend_adx','trend_adx_pos','trend_adx_neg','trend_adx_ind','trend_vortex_ind_pos',
'trend_vortex_ind_neg','trend_vortex_diff','trend_trix','trend_mass_index','trend_cci','trend_dpo','trend_kst',
'trend_kst_sig','trend_kst_diff','trend_ichimoku_a','trend_ichimoku_b','momentum_rsi','momentum_mfi','momentum_tsi',
'momentum_uo','momentum_stoch','momentum_stoch_signal','momentum_wr','momentum_ao']
# 'others_dr','others_cr'

# contin_vars = [base_vars+techincal_indicators]

n = len(train); n

test = test.set_index(index)
train = train.set_index(index)

len(contin_vars)


# In[489]:


train = train[cat_vars+contin_vars+[dep]].copy()
# , index


# In[490]:


# test[dep] = 0 
test = test[cat_vars+contin_vars+[dep]].copy()
# , index


# In[491]:


for v in cat_vars: train[v] = train[v].astype('category').cat.as_ordered()
#     todo: maybe change dep variable to category here for multiclass option


# In[492]:


apply_cats(test, train)
# test


# In[493]:


for v in contin_vars:
    train[v] = train[v].astype('float32')
    test[v] = test[v].astype('float32')


# We can now process our data...

# In[494]:


df, y, nas, mapper = proc_df(train, dep, do_scale=True)


# In[495]:


y.shape


# In[496]:


df_test, _, nas, mapper = proc_df(test, dep, do_scale=True, mapper=mapper, na_dict=nas)
train.head(30).T.head(70)


# In[497]:


nas={}


# In[498]:


df.head(2)


# In[499]:


df_test.head(2)


# Rake the last x% of rows as our validation set.

# In[500]:


train_size = int(n * trainRatio); train_size
val_idx = list(range(train_size, len(df)))
#val_idx = list(range(0, len(df)-train_size))
#val_idx = get_cv_idxs(n, val_pct=0.1)


# In[501]:


len(val_idx)


# ## DL

# We're ready to put together our models.

# We can create a ModelData object directly from our data frame. Is_Reg is set to False to turn this into a classification problem (from a regression).  Is_multi is set True because there there are three labels for target BUY,HOLD,SELL

# In[502]:


md = ColumnarModelData.from_data_frame(PATH, val_idx, df, y.astype('int'), cat_flds=cat_vars, bs=64,
                                      is_reg=False,is_multi=False,test_df=df_test)


# Some categorical variables have a lot more levels than others.

# In[503]:


cat_sz = [(c, len(train[c].cat.categories)+1) for c in cat_vars]


# In[504]:


cat_sz


# We use the *cardinality* of each variable (that is, its number of unique values) to decide how large to make its *embeddings*. Each level will be associated with a vector with length defined as below.

# In[505]:


emb_szs = [(c, min(50, (c+1)//2)) for _,c in cat_sz]


# In[506]:


emb_szs


# Check if cuda is available

# In[507]:


torch.cuda.is_available()


# In[508]:


len(df.columns)-len(cat_vars)


# In[509]:


m = md.get_learner(emb_szs, len(df.columns)-len(cat_vars),dropout, 2, [100,50], [0.03,0.06],None,True)


# In[510]:


m


# In[511]:


m.lr_find()
m.sched.plot(100)


# In[512]:


m.fit(lr, 3)


# In[513]:


m.fit(lr, 5, cycle_len=1)


# In[514]:


m.fit(lr, 3, cycle_len=4, cycle_mult=2 )


# In[515]:


m.save(modelName)


# In[516]:


m.load(modelName)


# ## Validation

# In[517]:


(x,y1)=m.predict_with_targs()


# Predicted vs Validation

# In[518]:


(np.argmax(x,axis=1),y1)


# In[519]:


y1.shape


# In[520]:


val = train.iloc[val_idx]
val[[dep]]
valpred = pd.DataFrame({'Close':val.Close,'index':val.index, 'action':val.action, 'predicted':np.argmax(x,axis=1)})[['Close','index', 'action','predicted']]
valpred.tail(100)


# Calculate the percent accuracy on the validation set

# In[521]:


calculateAccuracy(valpred)


# In[522]:


newdf,result = calculateNetProfit(valpred, 10000)
result


# In[523]:


newdf.head(10)


# In[524]:


newdf.plot(x='index', y=['Close', 'buyAmount'], style='o',figsize=(10,5), grid=True)


# In[525]:


newdf.tail(10)


# ## Test

# In[526]:


np.argmax(m.predict(True), axis =1)


# In[527]:


testPred = pd.DataFrame({'Timestamp':test.index, 'Close':test.Close, 'action':test.action, 'predicted':np.argmax(m.predict(True), axis =1)})[['Close','Timestamp', 'action', 'predicted']]
testPred.head(10)


# Calculate the percent accuracy on the test set

# In[528]:


calculateAccuracy(testPred)


# In[529]:


newdf,result = calculateNetProfit(testPred, 10000)
result


# In[530]:


newdf.head(10)


# In[531]:


newdf.tail(10)


# In[532]:


newdf.plot(x='Timestamp', y=['Close', 'buyAmount'], style='o',figsize=(10,5), grid=True)


# In[533]:


# csv_fn=f'{PATH}/tmp/sub4.csv'
# sub.to_csv(csv_fn, index=False)
# FileLink(csv_fn)


# ## Random Forest

# In[534]:


from sklearn.ensemble import RandomForestRegressor


# In[535]:


((val,trn), (y_val,y_trn)) = split_by_idx(val_idx, df.values, y)


# In[536]:


m = RandomForestRegressor(n_estimators=40, max_features=0.99, min_samples_leaf=2,
                          n_jobs=-1, oob_score=True)
m.fit(trn, y_trn);


# In[537]:


def PredtoClass(a):
    pred_class = []
    for i in range(len(a)):
        if a[i]<.5:
            pred_class.append(0)
        else:
            pred_class.append(1)
    return pred_class
def accuracy(preds, y_val):
    return  sum(1- abs(PredtoClass(preds) - y_val))/len(y_val)


# Accuracy on the validation set using a Random Forest Regressor

# In[538]:


preds = m.predict(val)
m.score(trn, y_trn), m.score(val, y_val), m.oob_score_, accuracy(preds, y_val)


# In[539]:


preds_test = m.predict(df_test.values)


# In[540]:


sub = pd.DataFrame({'Timestamp':test.index, 'action':PredtoClass(preds_test)})[['Timestamp', 'action']]
sub.head(10)


# In[541]:


# csv_fn=f'{PATH}/tmp/RFsub5.csv'
# sub.to_csv(csv_fn, index=False)
# FileLink(csv_fn)

