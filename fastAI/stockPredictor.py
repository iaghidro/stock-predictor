
# coding: utf-8

# # Structured and time series data

# 
# 
# 

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


from fastai.structured import *
from fastai.column_data import *
np.set_printoptions(threshold=50, edgeitems=20)

PATH='data/stock/'


# ## Create datasets

# preprocess data

# In[3]:


def concat_csvs(dirname):
    path = f'{PATH}{dirname}'
    filenames=glob(f"{PATH}/*.csv")

    wrote_header = False
    with open(f"{path}.csv","w") as outputfile:
        for filename in filenames:
            name = filename.split(".")[0]
            with open(filename) as f:
                line = f.readline()
                if not wrote_header:
                    wrote_header = True
                    outputfile.write("file,"+line)
                for line in f:
                     outputfile.write(name + "," + line)
                outputfile.write("\n")


# Feature Space:
# * train: Training set provided by competition
# * googletrend: trend of ethusd
# * test: testing set

# In[4]:


table_names = ['train']
#, 'test'
str('{PATH}{fname}.csv')


# We'll be using the popular data manipulation framework `pandas`. Among other things, pandas allows you to manipulate tables/data frames in python as one would in a database.
# 
# We're going to go ahead and load all of our csv's as dataframes into the list `tables`.

# In[5]:



tables = [pd.read_csv(f'{PATH}{fname}.csv', low_memory=False) for fname in table_names]


# In[6]:


from IPython.display import HTML


# We can use `head()` to get a quick look at the contents of each table:
# * train: Contains store information on a daily basis, tracks things like sales, customers, whether that day was a holdiay, etc.
# * store: general info about the store including competition, etc.
# * store_states: maps store to state it is in
# * state_names: Maps state abbreviations to names
# * googletrend: trend data for particular week/state
# * weather: weather conditions for each state
# * test: Same as training table, w/o sales and customers
# 

# In[7]:


for t in tables: display(t.head())


# This is very representative of a typical industry dataset.
# 
# The following returns summarized aggregate information to each table accross each field.

# In[8]:


for t in tables: display(DataFrameSummary(t).summary())


# ## Data Cleaning / Feature Engineering

# In[9]:


train = tables[0]


# In[10]:


len(train)


# In[11]:


def join_df(left, right, left_on, right_on=None, suffix='_y'):
    if right_on is None: right_on = left_on
    return left.merge(right, how='left', left_on=left_on, right_on=right_on, 
                      suffixes=("", suffix))


# delete unused

# In[12]:



train.drop('second',1,inplace=True)
train.drop('minute',1,inplace=True)
train.drop('action',1,inplace=True)
#train.drop('timePeriodStart',1,inplace=True)


# In[13]:


df = train
df.head()


# It's usually a good idea to back up large tables of extracted / wrangled features before you join them onto another one, that way you can go back to it easily if you need to make changes to it.

# In[14]:


df.columns


# In[15]:


joined = df #join_df(joined, df, ['timePeriodStart'])


# ## Durations

# We're going to set the active index to Date.

# In[16]:


df = df.set_index("timePeriodStart")
df.head()


# It is common when working with time series data to extract data that explains relationships across rows as opposed to columns, e.g.:
# * Running averages
# * Time until next event
# * Time since last event
# 
# This is often difficult to do with most table manipulation frameworks, since they are designed to work with relationships across columns. As such, we've created a class to handle this type of data.
# 
# We'll define a function `get_elapsed` for cumulative counting across a sorted dataframe. Given a particular field `fld` to monitor, this function will start tracking time since the last occurrence of that field. When the field is seen again, the counter is set to zero.
# 
# Upon initialization, this will result in datetime na's until the field is encountered. This is reset every time a new store is seen. We'll see how to use this shortly.

# In[17]:


df.reset_index(inplace=True)
#joined_test.reset_index(inplace=True)
df.head()


# In[18]:


#convert to date objects
df["timePeriodStart"] = pd.to_datetime(df.timePeriodStart)


# In[19]:


df.columns


# remove all 0 values 

# In[20]:


joined = joined[joined.priceClose!=0]


# In[21]:


joined.reset_index(inplace=True)


# ## Create features
# 

# Delete unused

# In[22]:


joined.head().T.head(40)


# Now that we've engineered all our features, we need to convert to input compatible with a neural network.
# 
# This includes converting categorical variables into contiguous integers or one-hot encodings, normalizing continuous features to standard normal, etc...

# In[23]:


cat_vars = ['hour', 'dayOfWeek', 'month', 'year']

contin_vars = ['priceOpen', 'tradesCount', 'volumeTraded', 'priceHigh', 'priceLow']

n = len(joined); n


# In[24]:


dep = 'priceClose'
joined = joined[cat_vars+contin_vars+[dep, 'timePeriodStart']].copy()


# In[25]:


#joined_test = joined[cat_vars+contin_vars+[dep, 'timePeriodStart']].copy()


# In[26]:


for v in cat_vars: joined[v] = joined[v].astype('category').cat.as_ordered()


# In[27]:


#apply_cats(joined_test, joined)


# In[28]:


for v in contin_vars:
    joined[v] = joined[v].fillna(0).astype('float32')
    #joined_test[v] = joined_test[v].fillna(0).astype('float32')


# To run on the full dataset, use this instead:

# In[29]:


samp_size = n
joined_samp = joined.set_index("timePeriodStart")


# In[30]:


joined_samp.head(2)


# In[31]:


df, y, nas, mapper = proc_df(joined_samp, 'priceClose', do_scale=True)
yl = np.log(y)


# In[32]:


#joined_test = joined_test.set_index("timePeriodStart")


# In[33]:


#df_test, _, nas, mapper = proc_df(joined_test, 'priceClose', do_scale=True, skip_flds=['Id'],
 #                                mapper=mapper, na_dict=nas)


# In[34]:


df.head()


# In time series data, cross-validation is not random. Instead, our holdout data is generally the most recent data, as it would be in real application. This issue is discussed in detail in [this post](http://www.fast.ai/2017/11/13/validation-sets/) on our web site.
# 
# One approach is to take the last 25% of rows (sorted by date) as our validation set.
# 
# An even better option for picking a validation set is using the exact same length of time period as the test set uses - this is implemented here:

# In[35]:


train_ratio = 0.75
# train_ratio = 0.9
train_size = int(samp_size * train_ratio); train_size
val_idx = list(range(train_size, len(df)))


# In[36]:


# TODO: SHOULD NOT HAVE TO DO THIS
# CONVERT INDEX TO DATETIME
df.index = pd.to_datetime(df.index)


# An even better option for picking a validation set is using the exact same length of time period as the test set uses - this is implemented here:

# In[37]:


#2018-04-03T23:53:00.726Z
val_idx = np.flatnonzero(
    (df.index<=datetime.datetime(2018,4,3)) & (df.index>=datetime.datetime(2018,3,1)))


# In[38]:


val_idx=[0]


# In[39]:


df.head().T.head(40)
# df.index[0]


# ## DL

# We're ready to put together our models.
# 
# Root-mean-squared percent error is the metric Kaggle used for this competition.

# In[40]:


def inv_y(a): return np.exp(a)

def exp_rmspe(y_pred, targ):
    targ = inv_y(targ)
    pct_var = (targ - inv_y(y_pred))/targ
    return math.sqrt((pct_var**2).mean())

max_log_y = np.max(yl)
y_range = (0, max_log_y*1.2)


# We can create a ModelData object directly from out data frame.

# In[41]:


md = ColumnarModelData.from_data_frame(PATH, val_idx, df, yl.astype(np.float32), cat_flds=cat_vars, bs=128)
# ,test_df=df_test
#todo: add a test set


# Some categorical variables have a lot more levels than others. Store, in particular, has over a thousand!

# In[42]:


cat_sz = [(c, len(joined_samp[c].cat.categories)+1) for c in cat_vars]


# In[43]:


cat_sz


# We use the *cardinality* of each variable (that is, its number of unique values) to decide how large to make its *embeddings*. Each level will be associated with a vector with length defined as below.

# In[44]:


emb_szs = [(c, min(50, (c+1)//2)) for _,c in cat_sz]


# In[45]:


emb_szs


# In[46]:


m = md.get_learner(emb_szs, len(df.columns)-len(cat_vars),
                   0.04, 1, [1000,500], [0.001,0.01], y_range=y_range)
lr = 1e-3


# In[47]:


m.lr_find()


# In[48]:


m.sched.plot(10)


# ### Sample

# In[49]:


m = md.get_learner(emb_szs, len(df.columns)-len(cat_vars),
                   0.04, 1, [1000,500], [0.001,0.01], y_range=y_range)
lr = 1e-3


# In[50]:


m.fit(lr, 3, metrics=[exp_rmspe])


# In[51]:


m.fit(lr, 5, metrics=[exp_rmspe], cycle_len=1)


# In[52]:


m.fit(lr, 2, metrics=[exp_rmspe], cycle_len=4)


# ### All

# In[53]:


m = md.get_learner(emb_szs, len(df.columns)-len(cat_vars),
                   0.04, 1, [1000,500], [0.001,0.01], y_range=y_range)
lr = 1e-3


# In[54]:


m.fit(lr, 1, metrics=[exp_rmspe])


# In[55]:


m.fit(lr, 3, metrics=[exp_rmspe])


# In[56]:


m.fit(lr, 3, metrics=[exp_rmspe], cycle_len=1)


# ### Test

# In[57]:


m = md.get_learner(emb_szs, len(df.columns)-len(cat_vars),
                   0.04, 1, [1000,500], [0.001,0.01], y_range=y_range)
lr = 1e-3


# In[58]:


m.fit(lr, 3, metrics=[exp_rmspe])


# In[59]:


m.fit(lr, 3, metrics=[exp_rmspe], cycle_len=1)


# In[60]:


m.save('val0')


# In[61]:


m.load('val0')


# In[62]:


x,y=m.predict_with_targs()


# In[63]:


exp_rmspe(x,y)


# In[64]:


#pred_test=m.predict(True)


# In[65]:


#pred_test = np.exp(pred_test)


# In[67]:


#joined_test['Sales']=pred_test


# In[68]:


csv_fn=f'{PATH}tmp/sub.csv'


# In[70]:


#joined_test[['Id','Sales']].to_csv(csv_fn, index=False)


# In[72]:


#FileLink(csv_fn)


# ## RF

# In[73]:


from sklearn.ensemble import RandomForestRegressor


# In[74]:


((val,trn), (y_val,y_trn)) = split_by_idx(val_idx, df.values, yl)


# In[75]:


m = RandomForestRegressor(n_estimators=40, max_features=0.99, min_samples_leaf=2,
                          n_jobs=-1, oob_score=True)
m.fit(trn, y_trn);


# In[76]:


preds = m.predict(val)
m.score(trn, y_trn), m.score(val, y_val), m.oob_score_, exp_rmspe(preds, y_val)

