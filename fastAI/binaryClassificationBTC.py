
# coding: utf-8

# # Structured data

# In[60]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[61]:


from fastai.structured import *
from fastai.column_data import *
np.set_printoptions(threshold=50, edgeitems=20)

PATH='data/stock/'


# ## Create datasets

# You can download the datasets used [here](https://www.kaggle.com/c/titanic/data).

# Feature Space:
# * train: Training set provided by competition
# * test: testing set

# In[62]:


table_names = ['btc-bitstamp-2012-01-01_to_2018-01-08']


# We'll be using the popular data manipulation framework `pandas`. Among other things, pandas allows you to manipulate tables/data frames in python as one would in a database.
# 
# We're going to go ahead and load all of our csv's as dataframes into the list `tables`.

# In[63]:


tables = [pd.read_csv(f'{PATH}{fname}.csv', low_memory=False) for fname in table_names]


# In[64]:


from IPython.display import HTML


# We can use `head()` to get a quick look at the contents of each table:
# * train: Contains Passenger info such as Gender, Age, Ticket, Fare and whether Survived, the prediction variable
# * test: Same as training table, w/o Survived
# 

# In[65]:


for t in tables: display(t.head())


# The following returns summarized aggregate information to each table accross each field.

# In[66]:


for t in tables: display(DataFrameSummary(t).summary())


# ## Data Cleaning / Feature Engineering

# As a structured data problem, we necessarily have to go through all the cleaning and feature engineering, even though we're using a neural network.

# In[67]:


train= tables[0]


# In[68]:


len(train)


# Time modifications

# In[69]:


#convert to date objects
train["Timestamp"] = pd.to_datetime(train["Timestamp"], unit='s')
train['hour'] = train.Timestamp.dt.hour;
train['minute'] = train.Timestamp.dt.minute;
train.head()


# In[70]:


#shift close prices forward
train['futureClose'] = train['Close'].shift(-1)


# In[71]:


#test
# testTrain = train[-5:]
# testTrain.apply(getTarget, axis=1)
# testTrain['action'] = (testTrain['futureClose'] > testTrain['Close'])
# testTrain = train[-5:]
# testTrain

# SET DEPENDENT VARIABLE ACTION
train['action'] = (train['futureClose'] > train['Close'])
train.action = train.action.astype(int)
train.head()


# convert the prediction variable to type integer

# In[72]:


train.action = train.action.astype(int)


# In[73]:


# May need to clean data and handle missing values

# add all date time values
add_datepart(train, "Timestamp", drop=False)

# edit columns

train["VolumeBTC"] = train["Volume_(BTC)"]
train.drop('Volume_(BTC)',1,inplace=True)
train["VolumeCurrency"] = train["Volume_(Currency)"]
train.drop('Volume_(Currency)',1,inplace=True)
train["WeightedPrice"] = train["Weighted_Price"]
train.drop('Weighted_Price',1,inplace=True)

# delete unused columns
train.drop('VolumeCurrency',1,inplace=True)
train.drop('futureClose',1,inplace=True)

train.reset_index(inplace=True)
train.head()


# remove all 0 values 

# In[74]:


train = train[train.Open!=0]
train = train[train.High!=0]
train = train[train.Low!=0]
train = train[train.Close!=0]
train = train[train.WeightedPrice!=0]


# In[75]:


# trim to a million records for now
# TODO: remove this
train = train[-100000:]
train.reset_index(inplace=True)


# In[76]:


train.to_feather(f'{PATH}train')


# We fill in missing values to avoid complications with `NA`'s. `NA` (not available) is how Pandas indicates missing values; many models have problems when missing values are present, so it's always important to think about how to deal with them. In these cases, we are picking an arbitrary *signal value* that doesn't otherwise appear in the data.

# ## Create features

# In[77]:


train = pd.read_feather(f'{PATH}train')


# In[78]:


train.head().T.head(40)


# In[79]:


display(DataFrameSummary(train).summary())


# Now that we've engineered all our features, we need to convert to input compatible with a neural network.
# 
# This includes converting categorical variables into contiguous integers or one-hot encodings, normalizing continuous features to standard normal, etc...

# In[80]:


train.head()


# Identify categorical vs continuous variables.  PassengerId serves as the unique identifier for each row.

# In[81]:


cat_vars = ['TimestampYear', 'TimestampMonth', 'TimestampWeek', 'TimestampDay', 'hour','minute', 'TimestampDayofweek', 'TimestampDayofyear', 
'TimestampIs_month_end', 'TimestampIs_month_start', 'TimestampIs_quarter_end', 'TimestampIs_quarter_start', 
'TimestampIs_year_end', 'TimestampIs_year_start', 'TimestampElapsed']

contin_vars = ['Open', 'Close','High', 'Low', 'VolumeBTC', 'WeightedPrice']

index='Timestamp'
dep = 'action'
n = len(train); n
   


# In[82]:


train = train[cat_vars+contin_vars+[dep, index]].copy()


# In[83]:


# test[dep] = 0
# test = test[cat_vars+contin_vars+[dep, index]].copy()


# In[84]:


for v in cat_vars: train[v] = train[v].astype('category').cat.as_ordered()


# In[85]:


# TODO: need to add this back
# apply_cats(test, train)


# In[86]:


for v in contin_vars:
    train[v] = train[v].astype('float32')
#     test[v] = test[v].astype('float32')


# In[87]:


samp_size = n
train_samp = train.set_index(index)
n


# We can now process our data...

# In[88]:


train_samp.head(2)


# In[89]:


df, y, nas, mapper = proc_df(train_samp, dep, do_scale=True)


# In[90]:


y.shape


# In[91]:


# df_test, _, nas, mapper = proc_df(test, dep, do_scale=True, 
#                                   mapper=mapper, na_dict=nas)


# For some reason, nas were found for Fare_log when there was not an NA value and it caused the code to fail downstream.  Here I inspected the value and then just removed the column :)

# In[92]:


nas
# df_test.Fare_log_na.unique()
# df_test.loc[df_test.Fare_log_na!=True]


# In[93]:


nas={}


# In[94]:


# df_test = df_test.drop(['Fare_log_na'], axis=1)


# In[95]:


df.head(2)


# In[96]:


# df_test.head(2)


# Rake the last 10% of rows as our validation set.

# In[97]:



train_ratio = 0.98
train_size = int(samp_size * train_ratio); train_size
val_idx = list(range(train_size, len(df)))
#val_idx = list(range(0, len(df)-train_size))
#val_idx = get_cv_idxs(n, val_pct=0.1)


# In[98]:


len(val_idx)


# ## DL

# We're ready to put together our models.

# We can create a ModelData object directly from our data frame. Is_Reg is set to False to turn this into a classification problem (from a regression).  Is_multi is set False because there is only one predicted label (Survived) per row (of type int).  

# In[99]:


md = ColumnarModelData.from_data_frame(PATH, val_idx, df, y.astype('int'), cat_flds=cat_vars, bs=64,
                                      is_reg=False,is_multi=False)
# ,test_df=df_test


# Some categorical variables have a lot more levels than others.

# In[100]:


cat_sz = [(c, len(train_samp[c].cat.categories)+1) for c in cat_vars]


# In[101]:


cat_sz


# We use the *cardinality* of each variable (that is, its number of unique values) to decide how large to make its *embeddings*. Each level will be associated with a vector with length defined as below.

# In[102]:


emb_szs = [(c, min(50, (c+1)//2)) for _,c in cat_sz]


# In[103]:


emb_szs


# Check if cude is available

# In[104]:


torch.cuda.is_available()


# In[105]:


len(df.columns)-len(cat_vars)


# In[106]:


m = md.get_learner(emb_szs, len(df.columns)-len(cat_vars),0.06, 2, [100,50], [0.03,0.06],None,True)


# In[107]:


m


# In[108]:


m.lr_find()
m.sched.plot(100)
lr = 1e-4


# In[109]:


m.fit(lr, 3)


# In[110]:


m.fit(lr, 5, cycle_len=1)


# In[111]:


m.fit(lr, 3, cycle_len=4, cycle_mult=2 )


# In[112]:


m.save('btcBinaryClassificationModel')


# In[113]:


m.load('btcBinaryClassificationModel')


# ## Validation

# In[114]:


(x,y1)=m.predict_with_targs()


# Predicted vs Validation

# In[115]:


(np.argmax(x,axis=1),y1)


# In[116]:


y1.shape


# In[117]:


val = train.iloc[val_idx]
val[[index,dep]]
valpred = pd.DataFrame({'Timestamp':val.Timestamp, 'action':val.action, 'predicted':np.argmax(x,axis=1)})[['Timestamp', 'action','predicted']]
valpred.head(10)


# Calculate the percent accuracy

# In[118]:


predicted = valpred.loc[valpred.action == valpred.predicted]
accuracy = len(predicted)/len(val)
accuracy


# ## Test and Kaggle Submission

# In[85]:


np.argmax(m.predict(True), axis =1)


# In[103]:


sub = pd.DataFrame({'Timestamp':test.index, 'action':np.argmax(m.predict(True), axis =1)})[['Timestamp', 'action']]
sub.head(10)


# In[87]:


csv_fn=f'{PATH}/tmp/sub4.csv'
sub.to_csv(csv_fn, index=False)
FileLink(csv_fn)


# ![image.png](attachment:image.png)
# 

# ## RF

# In[108]:


from sklearn.ensemble import RandomForestRegressor


# In[109]:


((val,trn), (y_val,y_trn)) = split_by_idx(val_idx, df.values, y)


# In[110]:


m = RandomForestRegressor(n_estimators=40, max_features=0.99, min_samples_leaf=2,
                          n_jobs=-1, oob_score=True)
m.fit(trn, y_trn);


# Accuracy of 87% on the validation set using a Random Forest Regressor.

# In[139]:


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


# In[141]:


preds = m.predict(val)
m.score(trn, y_trn), m.score(val, y_val), m.oob_score_, accuracy(preds, y_val)


# In[142]:


preds_test = m.predict(df_test.values)


# In[146]:


sub = pd.DataFrame({'PassengerId':test.index, 'Survived':PredtoClass(preds_test)})[['PassengerId', 'Survived']]
sub.head(10)


# In[147]:


csv_fn=f'{PATH}/tmp/RFsub5.csv'
sub.to_csv(csv_fn, index=False)
FileLink(csv_fn)


# This random forest submission also received a score of 0.77033, exactly the same as the nn score, despite the 86.7% validation set accuracy.