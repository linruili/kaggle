import pandas as pd
from sklearn.linear_model import LinearRegression
import time
import math
pd.set_option('display.width', 320)

#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#
#把test的prior的数据拿来作训练
merged_prior_test = pd.read_csv('../result/merged_prior_test.csv')
train_2 = merged_prior_test[merged_prior_test['max_order_num'] == merged_prior_test['order_number']]
train_2 = train_2[['user_id', 'order_id', 'product_id', 'add_to_cart_order', 'reordered']]

merged_prior_2 = merged_prior_test[merged_prior_test['max_order_num'] != merged_prior_test['order_number']]
train_2.to_csv('../result/train_2.csv', index=False)
merged_prior_2.to_csv('../result/merged_prior_2.csv', index=False)
print(train_2)
#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#
#统计在train里，但之前只购买过一次的商品，是在倒数第几次购物购买的
train = pd.read_csv('../result/train.csv')
features = pd.read_csv('../result/features.csv')
features = features[features['reordered']==1]
features = features[['user_id','product_id']]
train = train[train['reordered']==1]
train = train[['user_id','product_id']]

features = features.reset_index(drop=True)
features['is_in_train'] = 1
train = train.reset_index(drop=True)

merged = pd.merge(train, features, how='left', sort=False).fillna(0)
merged = merged[merged['is_in_train']==0]  #在train里，但之前只购买过一次的商品

prior = pd.read_csv('../result/merged_prior.csv')
merged = pd.merge(merged, prior[['user_id','product_id','order_number','max_order_num']])
merged['which_order'] = merged['order_number'] - merged['max_order_num'] - 1
print(merged['which_order'].value_counts())

#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#
#线性回归预测用户下一次会买多少件商品
def f1(arr):
    linreg = LinearRegression()
    linreg.fit(arr['order_number'].values.reshape(-1,1), arr['add_to_cart_order'].values.reshape(-1,1))
    order_num = arr['order_number'].values
    y = linreg.predict(order_num[-1]+1)
    return math.floor(y[0][0])

prior = pd.read_csv('../result/merged_prior.csv')
product_num = prior.groupby(['user_id','order_number'])['add_to_cart_order'].max().reset_index()
product_num = product_num.groupby('user_id').apply(f1)
product_num = pd.DataFrame(product_num, columns=['product_num']).reset_index()
print(product_num)
#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#
#预测下次购买时间
def f2(arr):
    linreg = LinearRegression()
    linreg.fit(arr['order_number'].values.reshape(-1,1), arr['days_since_prior_order'].values.reshape(-1,1))
    order_num = arr['order_number'].values
    y = linreg.predict(order_num[-1]+1)
    return round(y[0][0])

proir = pd.read_csv('../result/merged_prior.csv')
prior = proir[['user_id', 'order_number', 'days_since_prior_order']].dropna()

prior = prior.drop_duplicates().reset_index(drop=True)
predict_day_gap = prior.groupby('user_id').apply(f2)
predict_day_gap = pd.DataFrame(predict_day_gap, columns=['predict_day_gap']).reset_index()

train = pd.read_csv('../data/orders.csv')
train = train[train['eval_set']=='train']
train = train[['user_id', 'days_since_prior_order']].reset_index(drop=True)

predict_day_gap = pd.merge(predict_day_gap, train)
predict_day_gap['dif'] = abs(predict_day_gap['predict_day_gap']-predict_day_gap['days_since_prior_order'])
print(predict_day_gap)
print(len(predict_day_gap[predict_day_gap['dif']<5]) / len(predict_day_gap))
#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#
#创造小样本给2nd
orders = pd.read_csv('../input/orders.csv', nrows=99998)
order_products__prior = pd.read_csv('../input/order_products__prior.csv')
order_products__train = pd.read_csv('../input/order_products__train.csv')

order_products__prior = order_products__prior[order_products__prior['order_id'].isin(orders['order_id'])]
order_products__train = order_products__train[order_products__train['order_id'].isin(orders['order_id'])]

# print(order_products__train)
orders.to_csv('../input/orders.csv', index=False)
order_products__train.to_csv('../input/order_products__train.csv', index=False)
order_products__prior.to_csv('../input/order_products__prior.csv', index=False)
