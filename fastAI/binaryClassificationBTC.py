
# coding: utf-8

# # BTC Predictor

# In[578]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[579]:


from fastai.structured import *
from fastai.column_data import *
np.set_printoptions(threshold=50, edgeitems=20)
from ta import *

PATH='data/stock/'


# ## Create datasets

# You can download the datasets used [here](https://www.kaggle.com/c/titanic/data).

# Feature Space:
# * train: Training set provided by competition
# * test: testing set

# In[580]:


table_names = ['btc-bitstamp-2012-01-01_to_2018-01-08']


# We'll be using the popular data manipulation framework `pandas`. Among other things, pandas allows you to manipulate tables/data frames in python as one would in a database.
# 
# We're going to go ahead and load all of our csv's as dataframes into the list `tables`.

# In[581]:


tables = [pd.read_csv(f'{PATH}{fname}.csv', low_memory=False) for fname in table_names]


# In[582]:


from IPython.display import HTML


# We can use `head()` to get a quick look at the contents of each table:
# * train: Contains Passenger info such as Gender, Age, Ticket, Fare and whether Survived, the prediction variable
# * test: Same as training table, w/o Survived
# 

# In[583]:


for t in tables: display(t.head())


# The following returns summarized aggregate information to each table accross each field.

# In[584]:


# for t in tables: display(DataFrameSummary(t).summary())


# ## Data Cleaning / Feature Engineering

# As a structured data problem, we necessarily have to go through all the cleaning and feature engineering, even though we're using a neural network.

# In[585]:


train= tables[0]


# In[586]:


len(train)


# Time modifications

# In[587]:


#convert to date objects
train["Timestamp"] = pd.to_datetime(train["Timestamp"], unit='s')
train['hour'] = train.Timestamp.dt.hour;
train['minute'] = train.Timestamp.dt.minute;
train.head()


# SET DEPENDENT VARIABLE ACTION

# In[588]:


lookahead = 60
percentIncrease = 1.005
train['action'] =  train['Close'].rolling(window=lookahead).max() > percentIncrease * train['Close']

# train['action'] = 0;
# train.loc[train['Close'].rolling(window=lookahead).max() > train['Close'], 'action'] = 1
# train.loc[train['Close'].rolling(window=lookahead).max() > percentIncrease * train['Close'], 'action'] = 2

train.action = train.action.astype(int)

# target count by category
len(train[train.action==2]),len(train[train.action==1]),len(train[train.action==0])


# In[589]:


# edit columns
train["VolumeBTC"] = train["Volume_(BTC)"]
train.drop('Volume_(BTC)',1,inplace=True)
train["VolumeCurrency"] = train["Volume_(Currency)"]
train.drop('Volume_(Currency)',1,inplace=True)
train["WeightedPrice"] = train["Weighted_Price"]
train.drop('Weighted_Price',1,inplace=True)

# delete unused columns
train.drop('VolumeCurrency',1,inplace=True)
train.head()


# In[590]:


# trim to x records for now
# TODO: remove this
train = train.tail(110000)
len(train)


# In[591]:


# remove all 0 values 
train = train[train.Open!=0]
train = train[train.High!=0]
train = train[train.Low!=0]
train = train[train.Close!=0]
train = train[train.WeightedPrice!=0]
train = train[train.VolumeBTC!=0]
len(train)


# In[ ]:


# add technical analysis
train = add_all_ta_features(train, "Open", "High", "Low", "Close", "VolumeBTC", fillna=False)


# In[ ]:


# add all date time values
add_datepart(train, "Timestamp", drop=False)


# Create test set

# In[ ]:


# todo: make this into a percentage instead of hardcoding the test set
test = train.tail(10000)
test.reset_index(inplace=True)
train = train.head(100000)
train.reset_index(inplace=True)
len(train),len(test)


# In[ ]:


train.to_feather(f'{PATH}train')
test.to_feather(f'{PATH}test')


# ## Create features

# In[ ]:


train = pd.read_feather(f'{PATH}train')
test = pd.read_feather(f'{PATH}test')


# In[ ]:


pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
train.tail(50).T.head(100)


# In[ ]:


# display(DataFrameSummary(train).summary())
# break break break now


# Now that we've engineered all our features, we need to convert to input compatible with a neural network.
# 
# This includes converting categorical variables into contiguous integers or one-hot encodings, normalizing continuous features to standard normal, etc...

# In[ ]:


train.head()


# Identify categorical vs continuous variables.  PassengerId serves as the unique identifier for each row.

# In[ ]:


cat_vars = ['TimestampYear', 'TimestampMonth', 'TimestampWeek', 'TimestampDay', 'hour','minute', 'TimestampDayofweek',
'TimestampDayofyear','TimestampIs_month_end', 'TimestampIs_month_start', 'TimestampIs_quarter_end',
'TimestampIs_quarter_start','TimestampIs_year_end', 'TimestampIs_year_start']

contin_vars = ['Open', 'Close','High', 'Low', 'VolumeBTC', 'WeightedPrice', 'TimestampElapsed',
'volume_adi','volume_obv','volume_obvm','volume_cmf','volume_nvi','volatility_bbh','volatility_bbl','volatility_atr',
'volatility_bbm','momentum_rsi','momentum_stoch','trend_ichimoku_a','trend_ichimoku_b',
'trend_mass_index','trend_macd','trend_macd_signal','trend_macd_diff','trend_ema_fast','trend_ema_slow',
'trend_adx','trend_adx_pos','trend_adx_neg','trend_adx_ind']

index='Timestamp'
dep = 'action'
n = len(train); n

test = test.set_index(index)
train = train.set_index(index)
   


# In[ ]:


train = train[cat_vars+contin_vars+[dep]].copy()
# , index


# In[ ]:


# test[dep] = 0 
test = test[cat_vars+contin_vars+[dep]].copy()
# , index


# In[ ]:


for v in cat_vars: train[v] = train[v].astype('category').cat.as_ordered()


# In[ ]:


apply_cats(test, train)
# test


# In[ ]:


for v in contin_vars:
    train[v] = train[v].astype('float32')
    test[v] = test[v].astype('float32')


# We can now process our data...

# In[ ]:


df, y, nas, mapper = proc_df(train, dep, do_scale=True)


# In[ ]:


y.shape


# In[ ]:


df_test, _, nas, mapper = proc_df(test, dep, do_scale=True, mapper=mapper, na_dict=nas)
train.head(30).T.head(70)


# In[ ]:


nas={}


# In[ ]:


df.head(2)


# In[ ]:


df_test.head(2)


# Rake the last x% of rows as our validation set.

# In[ ]:


train_ratio = 0.9
train_size = int(n * train_ratio); train_size
val_idx = list(range(train_size, len(df)))
#val_idx = list(range(0, len(df)-train_size))
#val_idx = get_cv_idxs(n, val_pct=0.1)


# In[ ]:


len(val_idx)


# ## DL

# We're ready to put together our models.

# We can create a ModelData object directly from our data frame. Is_Reg is set to False to turn this into a classification problem (from a regression).  Is_multi is set True because there there are three labels for target BUY,HOLD,SELL

# In[ ]:


md = ColumnarModelData.from_data_frame(PATH, val_idx, df, y.astype('int'), cat_flds=cat_vars, bs=64,
                                      is_reg=False,is_multi=False,test_df=df_test)


# Some categorical variables have a lot more levels than others.

# In[ ]:


cat_sz = [(c, len(train[c].cat.categories)+1) for c in cat_vars]


# In[ ]:


cat_sz


# We use the *cardinality* of each variable (that is, its number of unique values) to decide how large to make its *embeddings*. Each level will be associated with a vector with length defined as below.

# In[ ]:


emb_szs = [(c, min(50, (c+1)//2)) for _,c in cat_sz]


# In[ ]:


emb_szs


# Check if cuda is available

# In[ ]:


torch.cuda.is_available()


# In[ ]:


len(df.columns)-len(cat_vars)


# In[ ]:


dropout = 0.06
m = md.get_learner(emb_szs, len(df.columns)-len(cat_vars),dropout, 2, [100,50], [0.03,0.06],None,True)


# In[ ]:


m


# In[ ]:


m.lr_find()
m.sched.plot(100)
lr = 1e-4


# In[ ]:


m.fit(lr, 3)


# In[ ]:


m.fit(lr, 5, cycle_len=1)


# In[ ]:


m.fit(lr, 3, cycle_len=4, cycle_mult=2 )


# In[ ]:


m.save('btcBinaryClassificationModel')


# In[ ]:


m.load('btcBinaryClassificationModel')


# ## Validation

# In[ ]:


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

# def calculateNetProfit(df, startAmount):
#     df.loc[df['Close'].rolling(window=lookahead).max() > df['Close'], 'action'] = 1
#     df.apply(lambda x: x**2)


# In[ ]:


(x,y1)=m.predict_with_targs()


# Predicted vs Validation

# In[ ]:


(np.argmax(x,axis=1),y1)


# In[ ]:


y1.shape


# In[ ]:


val = train.iloc[val_idx]
val[[dep]]
valpred = pd.DataFrame({'Close':val.Close,'index':val.index, 'action':val.action, 'predicted':np.argmax(x,axis=1)})[['Close','index', 'action','predicted']]
valpred.tail(10)


# Calculate the percent accuracy on the validation set

# In[ ]:


calculateAccuracy(valpred)


# ## Test

# In[ ]:


np.argmax(m.predict(True), axis =1)


# In[ ]:


testPred = pd.DataFrame({'Timestamp':test.index, 'Close':test.Close, 'action':test.action, 'predicted':np.argmax(m.predict(True), axis =1)})[['Close','Timestamp', 'action', 'predicted']]
testPred.head(10)


# Calculate the percent accuracy on the test set

# In[ ]:


calculateAccuracy(testPred)


# In[ ]:


# csv_fn=f'{PATH}/tmp/sub4.csv'
# sub.to_csv(csv_fn, index=False)
# FileLink(csv_fn)


# ## Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


((val,trn), (y_val,y_trn)) = split_by_idx(val_idx, df.values, y)


# In[ ]:


m = RandomForestRegressor(n_estimators=40, max_features=0.99, min_samples_leaf=2,
                          n_jobs=-1, oob_score=True)
m.fit(trn, y_trn);


# In[ ]:


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

# In[ ]:


preds = m.predict(val)
m.score(trn, y_trn), m.score(val, y_val), m.oob_score_, accuracy(preds, y_val)


# In[ ]:


preds_test = m.predict(df_test.values)


# In[ ]:


sub = pd.DataFrame({'Timestamp':test.index, 'action':PredtoClass(preds_test)})[['Timestamp', 'action']]
sub.head(10)


# In[ ]:


# csv_fn=f'{PATH}/tmp/RFsub5.csv'
# sub.to_csv(csv_fn, index=False)
# FileLink(csv_fn)

