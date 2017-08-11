import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import time
import math
pd.set_option('display.width', 320)

prior = pd.read_csv('../result/merged_prior.csv', nrows=1000)
order_num_ratio = prior.groupby(['user_id','product_id'])[['order_number']].sum().reset_index()
order_num_ratio = order_num_ratio.rename(columns={'order_number':'order_num_ratio'})

max_order_num = prior[['user_id','product_id','max_order_num']]
max_order_num = max_order_num.drop_duplicates().reset_index(drop=True)
order_num_ratio['order_num_ratio'] = order_num_ratio['order_num_ratio']/(max_order_num['max_order_num']**2)

time_ratio = prior.groupby(['user_id','product_id'])[['cumsum_time']].sum().reset_index()
time_ratio = time_ratio.rename(columns={'cumsum_time':'time_ratio'})
total_time = prior[['user_id','product_id','total_time']]
total_time = total_time.drop_duplicates().reset_index(drop=True)
time_ratio['time_ratio'] = time_ratio['time_ratio']/(total_time['total_time']**2)

features = pd.merge(order_num_ratio, time_ratio)

train = pd.read_csv('../result/train.csv')
train = train[train['reordered']==1]
train = train.rename(columns={'reordered':'reordered_in_train'})
train = train.drop(['order_id','add_to_cart_order'], axis=1)

features = pd.merge(features, train, how='left').fillna(0)
# prior = prior[prior['reordered']==1]
type = prior[['user_id','product_id','reordered']].rename(columns={'reordered':'type'})
type = type.groupby(['user_id','product_id'])['type'].sum().reset_index()
type.loc[type['type']==0, 'type'] = 'B'
type.loc[type['type']!='B', 'type'] = 'A'
features = pd.merge(features, type, sort=False)

last_order_gap = prior.groupby(['user_id', 'product_id'])[['order_number']].max().reset_index()\
    .rename(columns={'order_number':'last_order_gap'})
last_order_gap = pd.merge(last_order_gap, max_order_num, sort=False)
last_order_gap['last_order_gap'] = last_order_gap['max_order_num'] - last_order_gap['last_order_gap']
last_order_gap = last_order_gap.drop(['max_order_num'], axis=1)

features = pd.merge(features, last_order_gap, sort=False)
print(features)