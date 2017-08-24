#train类的用户prior统计
import pandas as pd

pd.set_option('display.width', 320)

#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#
#创建train.csv
orders = pd.read_csv('../data/orders.csv')
orders = orders[['user_id', 'order_id']]
order_products_train = pd.read_csv('../data/order_products__train.csv')
train = pd.merge(orders, order_products_train)
train.to_csv('../result/train.csv', index=False)
#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#
#train的feature
# orders = pd.read_csv('../data/orders.csv')
# order_products__prior = pd.read_csv('../data/order_products__prior.csv')
# test = orders[orders['eval_set'] == 'test']
# other = ~orders['user_id'].isin(test['user_id'])
# prior = orders[other]
# prior = prior[orders['eval_set']!='train']
#
# max_order_num = prior.groupby('user_id')[['order_number']].max()
# max_order_num = max_order_num.rename(columns={'order_number':'max_order_num'})
# prior = pd.merge(prior, max_order_num, left_on='user_id', right_index=True)
#
# cumsum_time = prior.groupby('user_id')[['days_since_prior_order']].cumsum().fillna(0)
# cumsum_time = cumsum_time.rename(columns={'days_since_prior_order':'cumsum_time'})
# prior = pd.merge(prior, cumsum_time, left_index=True, right_index=True)
#
# total_time = prior.groupby('user_id')[['days_since_prior_order']].sum()
# total_time = total_time.rename(columns={'days_since_prior_order':'total_time'})
# prior = pd.merge(prior, total_time, left_on='user_id', right_index=True)
#
# prior = pd.merge(prior, order_products__prior)
# prior.to_csv('../result/merged_prior.csv',index=False)


#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#
#test的feature
orders = pd.read_csv('../data/orders.csv')
order_products__prior = pd.read_csv('../data/order_products__prior.csv')
test = orders[orders['eval_set'] == 'test']
test_case = orders['user_id'].isin(test['user_id'])
prior = orders[test_case]
prior = prior[orders['eval_set']!='test']

max_order_num = prior.groupby('user_id')[['order_number']].max()
max_order_num = max_order_num.rename(columns={'order_number':'max_order_num'})
prior = pd.merge(prior, max_order_num, left_on='user_id', right_index=True)

cumsum_time = prior.groupby('user_id')[['days_since_prior_order']].cumsum().fillna(0)
cumsum_time = cumsum_time.rename(columns={'days_since_prior_order':'cumsum_time'})
prior = pd.merge(prior, cumsum_time, left_index=True, right_index=True)

total_time = prior.groupby('user_id')[['days_since_prior_order']].sum()
total_time = total_time.rename(columns={'days_since_prior_order':'total_time'})
prior = pd.merge(prior, total_time, left_on='user_id', right_index=True)

prior = pd.merge(prior, order_products__prior)
prior.to_csv('../result/merged_prior_test.csv',index=False)