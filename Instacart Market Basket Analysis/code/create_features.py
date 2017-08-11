
import pandas as pd

pd.set_option('display.width', 320)
#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#
#两个feature：order_num_ratio、time_ratio、last_order_gap
#last_order_gap = max_order_num - product_max_order_num
prior = pd.read_csv('../result/merged_prior.csv')
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

type = prior[['user_id','product_id','reordered']].rename(columns={'reordered':'type'})
type = type.groupby(['user_id','product_id'])['type'].sum().reset_index()
type.loc[type['type']==0, 'type'] = 'B'
type.loc[type['type']!='B', 'type'] = 'A'
features = pd.merge(features, type, sort=False)
features = features.iloc[:,[0,1,4,5,2,3]]

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

order_products_prior = order_products_prior[order_products_prior['reordered']==1]
rebuy_total_sale = order_products_prior['product_id'].value_counts()
rebuy_total_sale = pd.DataFrame({'product_id':rebuy_total_sale.index, 'rebuy_total_sale':rebuy_total_sale.values},\
                                index=range(len(rebuy_total_sale)))

sales = pd.merge(total_sale, rebuy_total_sale, how='outer').fillna(0).sort_values(by='product_id')
# sales.to_csv('../result/sales.csv', index=False)

features = pd.merge(features, sales, sort=False)
features = features.sort_values(by=['user_id','product_id'])
features.to_csv('../result/features.csv', index=False)

