import pandas as pd
from sklearn.linear_model import LinearRegression
import time
import math
pd.set_option('display.width', 320)

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
#
