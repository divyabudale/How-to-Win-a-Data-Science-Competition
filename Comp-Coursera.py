
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from itertools import product
import seaborn as sns
import xgboost as xgb
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import gc
import time
import pickle
from tqdm import tqdm_notebook
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import dask_ml.joblib
from sklearn.externals.joblib import parallel_backend
from sklearn.feature_extraction.text import TfidfVectorizer


# In[2]:


pd.set_option('display.max_rows', 600)
pd.set_option('display.max_columns', 50)

def downcast_dtypes(df):
    '''
        Changes column types in the dataframe: 
                
                `float64` type to `float32`
                `int64`   type to `int32`
    '''
    
    # Select columns to downcast
    float_cols_64 = [c for c in df if df[c].dtype == "float64"]
    float_cols_32 = [c for c in df if df[c].dtype == "float32"]
    int_cols_64 =   [c for c in df if df[c].dtype == "int64"]
    int_cols_32 =   [c for c in df if df[c].dtype == "int32"]
    
    # Downcast
    df[float_cols_64] = df[float_cols_64].astype(np.float16)
    df[int_cols_64]   = df[int_cols_64].astype(np.int16)
    df[float_cols_32] = df[float_cols_32].astype(np.float16)
    df[int_cols_32]   = df[int_cols_32].astype(np.int16)
    
    
    return df


# In[3]:


ts = time.time()
#loading the data from the files
train           = pd.read_csv('C:/Users/divya/Downloads/all/sales_train.csv/sales_train_v2.csv')
items           = pd.read_csv('C:/Users/divya/Downloads/all/items.csv')
item_cat        = pd.read_csv('C:/Users/divya/Downloads/all/item_categories.csv')
shops           = pd.read_csv('C:/Users/divya/Downloads/all/shops.csv')
test            = pd.read_csv('C:/Users/divya/Downloads/all/test.csv/test.csv')
time.time()-ts


# In[4]:


plt.figure(figsize=(10,4))
plt.xlim(-100, 3000)
sns.boxplot(x=train.item_cnt_day)

plt.figure(figsize=(10,4))
plt.xlim(train.item_price.min(), train.item_price.max()*1.1)
sns.boxplot(x=train.item_price)


# In[5]:


#remove the outliers
train = train[train.item_price<100000]
train = train[train.item_cnt_day<1001]
#median = train[(train.shop_id==32)&(train.item_id==2973)&(train.date_block_num==4)&(train.item_price>0)].item_price.median()
#train.loc[train.item_price<0, 'item_price'] = median


# In[6]:


#create a table with all possible combination for shop id and item id
index_cols = ['shop_id', 'item_id', 'date_block_num']
# For every month we create a grid from all shops/items combinations from that month
grid = [] 
for block_num in train['date_block_num'].unique():
    curr_shops = train[train['date_block_num']==block_num].shop_id.unique()
    curr_items = train[train['date_block_num']==block_num].item_id.unique()
    grid.append(np.array(list(product(*[curr_shops, curr_items, [block_num]])),dtype='int32'))
#turn the grid into pandas dataframe
grid = pd.DataFrame(np.vstack(grid), columns = index_cols,dtype=np.int32)
#get aggregated values for (shop_id, item_id, month)
gb = train.groupby(index_cols,as_index=False).agg({'item_cnt_day':'sum'})
#fix column names
gb = gb.rename(columns={'item_cnt_day':'target'})
#join aggregated data to the grid
all_data = pd.merge(grid,gb,how='left',on=index_cols).fillna(0)
#sort the data
#train.sort_values(['date_block_num','shop_id','item_id'],inplace=True)
del gb, grid, curr_shops, curr_items


# In[7]:


all_data['target'] = (all_data['target']
                                .fillna(0)
                                .clip(0,20) # NB clip target here
                                .astype(np.float32))


# In[8]:


#create test dataset
test['date_block_num']=34
test['target']=np.nan
test=test.drop(['ID'],axis=1)


# In[9]:


#merge train and test
#all_data['set'] = 'train'
#test['set'] = 'test'
all_data = pd.concat([all_data,test])


# In[10]:


all_data.shape


# In[11]:


#calculate prev month target for shop id and item id
prev_month = all_data[['date_block_num','shop_id','item_id','target']] 
#prev_month['date_block_num']=  prev_month['date_block_num']+ 1
prev_month.loc[:,'date_block_num']+=  1
prev_month = prev_month.rename(columns={'target':'target_prev_month'}) 
all_data = pd.merge(all_data, prev_month, how='left', on=['date_block_num','shop_id','item_id']).fillna(0.)
del prev_month


# In[12]:


#calculate target for item id
gb = all_data.groupby(['date_block_num','item_id'],as_index=False).agg({'target':'sum'})
gb = gb.rename(columns={'target':'item_target'})
# Join it to the all_data
all_data = pd.merge(all_data, gb, how='left', on=['item_id', 'date_block_num']).fillna(0)
del gb


# In[13]:


#calculate prev month target for item id
prev_month = all_data[['date_block_num','item_id','item_target']] .drop_duplicates()
prev_month.loc[:,'date_block_num'] += 1
prev_month = prev_month.rename(columns={'item_target':'item_target_prev_month'}) 
all_data = pd.merge(all_data, prev_month, how='left', on=['date_block_num','item_id']).fillna(0.)
del prev_month
#all_data.head()


# In[14]:


#calculate target for shop id
gb = all_data.groupby(['date_block_num','shop_id'],as_index=False).agg({'target':'sum'})
gb = gb.rename(columns={'target':'shop_target'})
# Join it to the all_data
all_data = pd.merge(all_data, gb, how='left', on=['shop_id', 'date_block_num']).fillna(0)
del gb


# In[15]:


#calculate prev month target for shop id
prev_month = all_data[['date_block_num','shop_id','shop_target']] .drop_duplicates()
prev_month.loc[:,'date_block_num'] += 1
prev_month = prev_month.rename(columns={'shop_target':'shop_target_prev_month'}) 
all_data = pd.merge(all_data, prev_month, how='left', on=['date_block_num','shop_id']).fillna(0.)
del prev_month
#all_data.head()


# In[16]:


#calculate avg item price for shop id and item id
gb = train.groupby(['date_block_num','item_id','shop_id'],as_index=False).agg({'item_price':'mean'})
gb = gb.rename(columns={'item_price':'avg_item_price'})
# Join it to the all_data
all_data = pd.merge(all_data, gb, how='left', on=['item_id', 'shop_id', 'date_block_num']).fillna(0)
del gb


# In[17]:


#calculate prev month avg item price for shop id and item id
prev_month = all_data[['date_block_num','shop_id','item_id','avg_item_price']] 
prev_month.loc[:,'date_block_num'] += 1
prev_month = prev_month.rename(columns={'avg_item_price':'avg_price_prev_month'}) 
all_data = pd.merge(all_data, prev_month, how='left', on=['date_block_num','shop_id','item_id']).fillna(0.)
del prev_month
#all_data.head()


# In[18]:


gb = all_data.groupby(['shop_id','item_id'])['date_block_num'].agg(['min','max']).rename(columns={'min':'first','max':'last'})
# max_date = all_data.date_block_num.max()
# Join it to the all_data
all_data = pd.merge(all_data, gb, how='left', on=['item_id', 'shop_id']).fillna(0)
#all_data['first_shop_item_sale'] = max_date - all_data['last']
all_data['first_shop_item_sale'] = all_data['date_block_num'] - all_data['first'] 
del gb


# In[19]:


gb = all_data.groupby(['item_id'])['date_block_num'].agg(['min','max']).rename(columns={'min':'first_1','max':'last_1'})
# max_date = all_data.date_block_num.max()
# Join it to the all_data
all_data = pd.merge(all_data, gb, how='left', on=['item_id']).fillna(0)
#all_data['first_item_sale'] = max_date - all_data['last']
all_data['first_item_sale'] = all_data['date_block_num'] - all_data['first_1'] 
del gb


# In[20]:


to_drop_cols = ['first','last','first_1','last_1','item_target','avg_item_price','shop_target']
all_data = all_data.drop(to_drop_cols, axis=1)


# In[21]:


#extract info from item name
items_subset = items[['item_id', 'item_name']]
features = 30
tfidf = TfidfVectorizer(max_features=features) 
items_df = pd.DataFrame(tfidf.fit_transform(items_subset['item_name']).toarray())
cols = items_df.columns
for i in range(features):
    feature_name = 'item_name_tfidf_' + str(i)
    items_subset[feature_name] = items_df[cols[i]]
    #new_features.append(feature_name)
items_subset.drop('item_name', axis = 1, inplace = True)
all_data = all_data.merge(items_subset, on = 'item_id', how = 'left')
all_data.head()


# In[22]:


all_data[(all_data['shop_id']==17)&(all_data['item_id']==30)].head(10)


# In[23]:


shift_range = [1, 2, 3, 4, 5, 6 ,7, 8, 9, 10, 11, 12]

#shift_range_1 = list(range(1,34))


# In[24]:


from sklearn.externals.joblib import Memory
memory = Memory(cachedir='/tmp', verbose=0)
@memory.cache
def lag_features1(all_data, shift_range, cols_to_rename):
    tmp = all_data[['date_block_num','shop_id','item_id',cols_to_rename]]
    for month_shift in (shift_range):
        train_shift = tmp.copy()
        train_shift.columns = ['date_block_num','shop_id','item_id', cols_to_rename+'_lag_'+str(month_shift)]
        train_shift['date_block_num'] = train_shift['date_block_num'] + month_shift
        all_data = pd.merge(all_data, train_shift, on=index_cols, how='left').fillna(0)
    return all_data


# In[25]:


ts = time.time()
#all_data = lag_features1(all_data, shift_range, 'target')
all_data = lag_features1(all_data, shift_range, 'target_prev_month')
#all_data = lag_features1(all_data, shift_range, 'avg_item_price')
all_data = lag_features1(all_data, shift_range, 'avg_price_prev_month')
#all_data = lag_features1(all_data, shift_range, 'first_shop_item_sale') #new
time.time()-ts


# In[26]:


from sklearn.externals.joblib import Memory
memory = Memory(cachedir='/tmp', verbose=0)
@memory.cache
def lag_features(all_data, shift_range, cols_to_rename):
    tmp = all_data[['date_block_num','item_id',cols_to_rename]].drop_duplicates()
    for month_shift in (shift_range):
        train_shift = tmp.copy()
        train_shift.columns = ['date_block_num','item_id', cols_to_rename+'_lag_'+str(month_shift)]
        train_shift['date_block_num'] = train_shift['date_block_num'] + month_shift
        all_data = pd.merge(all_data, train_shift, on=['date_block_num','item_id'], how='left').fillna(0)
    return all_data


# In[27]:


ts = time.time()
all_data = lag_features(all_data, shift_range, 'item_target_prev_month')
#all_data = lag_features(all_data, shift_range, 'itemid_price') 
time.time()-ts


# In[28]:


from sklearn.externals.joblib import Memory
memory = Memory(cachedir='/tmp', verbose=0)
@memory.cache
def lag_features(all_data, shift_range, cols_to_rename):
    tmp = all_data[['date_block_num','shop_id',cols_to_rename]].drop_duplicates()
    for month_shift in (shift_range):
        train_shift = tmp.copy()
        train_shift.columns = ['date_block_num','shop_id', cols_to_rename+'_lag_'+str(month_shift)]
        train_shift['date_block_num'] = train_shift['date_block_num'] + month_shift
        all_data = pd.merge(all_data, train_shift, on=['date_block_num','shop_id'], how='left').fillna(0)
    return all_data


# In[29]:


ts = time.time()
all_data = lag_features(all_data, shift_range, 'shop_target_prev_month')
#all_data = lag_features(all_data, shift_range, 'itemid_price_prev_month')
time.time()-ts


# In[30]:


from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=False)
column = "item_id"
encoded_column = column + "_mean_target"
train_new = pd.DataFrame(index=all_data.index, columns=all_data.columns)
train_new[encoded_column] = np.nan

for train_index, val_index in kf.split(all_data):
    #split into train and validation
    x_train , x_val = all_data.iloc[train_index], all_data.iloc[val_index]
    #estimate the encoding on train and map it to validation
    means = x_val[column].map(x_train.groupby(column).target.mean())
    #apply the encoding to the validation
    x_val[encoded_column] = means
    # train_new is a dataframe copy we made of the training data
    train_new.iloc[val_index] = x_val

train_new.fillna(0.3502, inplace=True)
#x = train_new.item_id_mean_target.mean()
#print(x)
#train_new.fillna(x, inplace=True)
train_new=train_new.drop(['item_id'],axis=1)
all_data=train_new    


# In[31]:


all_data = downcast_dtypes(all_data)
gc.collect()
all_data.info(memory_usage='deep')


# In[32]:


all_data.head()
#all_data[(all_data['shop_id']==59)&(all_data['item_id']==2574)].head(10)


# In[33]:


to_drop_cols = ['target']#,'avg_price_prev_month','item_cnt_prev_month']
to_drop_cols


# In[34]:


#train_df=all_data[all_data['date_block_num']<33]
#train_df= train_df.sample(n=100000,replace=True)
#X_train = train_df.drop(to_drop_cols, axis=1)
#Y_train = train_df['target']

X_train = all_data[all_data.date_block_num < 33].drop(to_drop_cols, axis=1)
Y_train = all_data[all_data.date_block_num < 33]['target']
#X_train = X_train.drop(['date_block_num'],axis=1)

X_valid = all_data[all_data.date_block_num == 33].drop(to_drop_cols, axis=1)
Y_valid = all_data[all_data.date_block_num == 33]['target']
#X_valid = X_valid.drop(['date_block_num'],axis=1)

X_test = all_data[all_data.date_block_num == 34].drop(to_drop_cols, axis=1)
#X_test = X_test.drop(['date_block_num'],axis=1)


# In[35]:


print('Train shape',X_train.shape)
print('Validation shape',X_valid.shape)
print('Test shape',X_test.shape)


# In[36]:


ts = time.time()

model = XGBRegressor(
    max_depth=11,
    n_estimators=2000,
    min_child_weight=0.5, 
    colsample_bytree=0.8, 
    subsample=1, 
    eta=0.3,    
    seed=42)

model.fit(
    X_train, 
    Y_train, 
    eval_metric="rmse", 
    eval_set=[(X_train, Y_train), (X_valid, Y_valid)], 
    verbose=True, 
    early_stopping_rounds = 10)

time.time() - ts


# In[37]:


print('Train MSE:', np.sqrt(mean_squared_error(Y_train.clip(0.,20.),model.predict(X_train).clip(0.,20.))))
print('Valid MSE:', np.sqrt(mean_squared_error(Y_valid.clip(0.,20.),model.predict(X_valid).clip(0.,20.))))


# In[40]:


#test_df = X_test.loc[:, test.columns != 'target']
y_pred = model.predict(X_test).clip(0., 20.)
y_pred


# In[41]:


preds = pd.DataFrame(y_pred, columns=['item_cnt_month'])
preds.to_csv('submission.csv',index_label='ID')

