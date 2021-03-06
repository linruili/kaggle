import pandas as pd

pd.set_option('display.width', 320)
#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#
#两个feature：order_num_ratio、time_ratio、last_order_gap
#last_order_gap = max_order_num - product_max_order_num
prior = pd.read_csv('../result/merged_prior_test.csv')
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

last_order_gap = prior.groupby(['user_id', 'product_id'])[['order_number']].max().reset_index()\
    .rename(columns={'order_number':'last_order_gap'})
last_order_gap = pd.merge(last_order_gap, max_order_num, sort=False)
last_order_gap['last_order_gap'] = last_order_gap['max_order_num'] - last_order_gap['last_order_gap']
last_order_gap = last_order_gap.drop(['max_order_num'], axis=1)
features = pd.merge(features, last_order_gap, sort=False)

#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#
#创建features：总销量和再次购买销量
order_products_prior = pd.read_csv('../data/order_products__prior.csv')
total_sale = order_products_prior['product_id'].value_counts()
total_sale = pd.DataFrame({'product_id':total_sale.index, 'total_sale':total_sale.values}, index=range(len(total_sale)))

order_products_prior_tmp = order_products_prior[order_products_prior['reordered']==1]
rebuy_total_sale = order_products_prior_tmp['product_id'].value_counts()
rebuy_total_sale = pd.DataFrame({'product_id':rebuy_total_sale.index, 'rebuy_total_sale':rebuy_total_sale.values},\
                                index=range(len(rebuy_total_sale)))

sales = pd.merge(total_sale, rebuy_total_sale, how='outer').fillna(0).sort_values(by='product_id')

sales['rebuy_product_ratio'] = sales['rebuy_total_sale']/sales['total_sale']
sales = sales.drop(['rebuy_total_sale','total_sale'], axis=1)

# sales.to_csv('../result/sales.csv', index=False)

features = pd.merge(features, sales, sort=False)
#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#
#features：总订单数max_order_num_average和加入购物车的顺序add_to_cart_average
tmp = prior[['user_id','order_number','product_id','max_order_num','add_to_cart_order']]
max_order_num_average = tmp[['user_id','max_order_num']].drop_duplicates().reset_index(drop=True)
max_order_num_average = max_order_num_average.groupby('user_id')['max_order_num'].mean().reset_index()\
    .rename(columns={'max_order_num':'max_order_num_average'})

add_to_cart_max = tmp.groupby(['user_id','order_number'])['add_to_cart_order'].max().reset_index()\
    .rename(columns={'add_to_cart_order':'add_to_cart_max'})
tmp = pd.merge(tmp, add_to_cart_max)
tmp['add_to_cart_ratio'] = tmp['add_to_cart_order']/tmp['add_to_cart_max']
add_to_cart_ratio_average = tmp.groupby(['user_id','product_id'])['add_to_cart_ratio'].mean().reset_index()\
    .rename(columns={'add_to_cart_ratio':'add_to_cart_ratio_average'})

features = pd.merge(features, add_to_cart_ratio_average)
features = pd.merge(features, max_order_num_average)

features = features.sort_values(by=['user_id','product_id'])

#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#
#features：rebuy_aisle_ratio、rebuy_depertment_ratio
products = pd.read_csv('../data/products.csv')
tmp = pd.merge(prior[['user_id','product_id', 'reordered']], products)
department_product_count = tmp.groupby('department_id')['product_id'].count().reset_index()\
    .rename(columns={'product_id':'department_product_count'})

aisle_product_count = tmp.groupby('aisle_id')['product_id'].count().reset_index()\
    .rename(columns={'product_id':'aisle_product_count'})

tmp = tmp[tmp['reordered']==1]
rebuy_depertment_count = tmp.groupby('department_id')['product_id'].count().reset_index()\
    .rename(columns={'product_id':'rebuy_depertment_count'})
rebuy_aisle_count = tmp.groupby('aisle_id')['product_id'].count().reset_index()\
    .rename(columns={'product_id':'rebuy_aisle_count'})

rebuy_depertment_ratio = pd.merge(department_product_count, rebuy_depertment_count)
rebuy_depertment_ratio['rebuy_depertment_ratio'] = \
    rebuy_depertment_ratio['rebuy_depertment_count']/rebuy_depertment_ratio['department_product_count']
rebuy_aisle_ratio = pd.merge(aisle_product_count, rebuy_aisle_count)
rebuy_aisle_ratio['rebuy_aisle_ratio'] = \
    rebuy_aisle_ratio['rebuy_aisle_count']/rebuy_aisle_ratio['aisle_product_count']

rebuy_depertment_ratio = rebuy_depertment_ratio[['department_id', 'rebuy_depertment_ratio']]
rebuy_aisle_ratio = rebuy_aisle_ratio[['aisle_id', 'rebuy_aisle_ratio']]

aisle_department_feature = pd.merge(products, rebuy_aisle_ratio)
aisle_department_feature = pd.merge(aisle_department_feature, rebuy_depertment_ratio)
aisle_department_feature = aisle_department_feature[['product_id', 'rebuy_aisle_ratio', 'rebuy_depertment_ratio']]

features = pd.merge(features, aisle_department_feature)

#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#
#创建features：order_hour_dif、order_dow_dif_sum
orders = pd.read_csv('../data/orders.csv')

def f3(arr):
    train_case = arr[arr['eval_set']=='test']
    arr['target_interval'] = train_case['order_hour_interval'].iloc[0]
    arr['target_order_dow'] = train_case['order_dow'].iloc[0]
    return arr

def f4(x):
    if(x==19 or x==1):
        x = 0.5
    elif(x==18 or x==2):
        x = 0.25
    elif(x==0):
        x=1
    else:
        x = 0
    return x

def f5(x):
    if x==0:
        x = 1
    elif x==1 or x==6:
        x = 0.25
    else:
        x = 0
    return x

test = orders[orders['eval_set'] == 'test']
test_cases = orders['user_id'].isin(test['user_id'])
prior_train = orders[test_cases]
prior_train['order_hour_interval'] = prior_train['order_hour_of_day'].map({\
    # 0:1,1:1,2:1,3:1,4:1,5:1,6:1,7:1,8:2,9:2,10:3,11:3,12:4,13:4,14:5,15:5,16:5,17:6,18:6,19:7,20:7,21:8,22:8,23:8})
    7:7,6:7,5:6,4:6,3:5,2:5,1:4,0:4,8:8,9:9,10:10,11:11,12:12,13:13,14:14,15:15,16:16,17:17,18:18,19:19,20:20,21:21,22:22,23:23})
prior_train = prior_train.groupby('user_id').apply(f3)
prior_train['order_hour_dif'] = abs(prior_train['target_interval'] - prior_train['order_hour_interval'])
prior_train = prior_train.drop(['order_hour_interval', 'target_interval', 'order_hour_of_day'], axis=1)
prior_train['order_hour_dif'] = prior_train['order_hour_dif'].apply(f4)

prior_train['order_dow_dif'] = abs(prior_train['target_order_dow'] - prior_train['order_dow'])
prior_train = prior_train.drop(['target_order_dow','order_dow'], axis=1)
prior_train['order_dow_dif'] = prior_train['order_dow_dif'].apply(f5)

prior_train = pd.merge(prior_train, order_products_prior[['order_id','product_id']])
order_hour_dif_sum = prior_train.groupby(['user_id', 'product_id'])['order_hour_dif'].sum().reset_index().\
    rename(columns={'order_hour_dif':'order_hour_dif_sum'})
order_dow_dif_sum = prior_train.groupby(['user_id', 'product_id'])['order_dow_dif'].sum().reset_index().\
    rename(columns={'order_dow_dif':'order_dow_dif_sum'})

hour_dow = pd.merge(order_hour_dif_sum, order_dow_dif_sum)
features = pd.merge(features, hour_dow)

features = features.sort_values(by=['user_id', 'product_id'])
features.to_csv('../result/features_test.csv', index=False)

a = features['user_id'].value_counts()
print(len(a))