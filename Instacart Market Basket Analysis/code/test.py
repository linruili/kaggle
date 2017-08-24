import pandas as pd
import numpy as np
from itertools import product
from sklearn.linear_model import LogisticRegression

pd.set_option('display.width', 320)
IDIR = '../data/'

df_train = pd.read_csv('../result/tmp_df_train.csv')
df_train = df_train.fillna(1)
df_test = pd.read_csv('../result/tmp_df_test.csv')
labels = np.load('../result/tmp_lables.npy')


f_to_use = ['user_total_orders', 'user_total_items', 'total_distinct_items',
            'user_average_days_between_orders', 'user_average_basket',
            'order_hour_of_day', 'days_since_prior_order', 'days_since_ratio',
            'aisle_id', 'department_id', 'product_orders', 'product_reorders',
            'product_reorder_rate', 'UP_orders', 'UP_orders_ratio',
            'UP_average_pos_in_cart', 'UP_reorder_rate', 'UP_orders_since_last',
            'UP_delta_hour_vs_last']  # 'dow', 'UP_same_dow_as_last_order'

print('LogisticRegression train')
logreg = LogisticRegression(C=0.2)
logreg.fit(df_train[f_to_use], labels)
del df_train

### build candidates list for test ###

print('LogisticRegression predict')
preds = logreg.predict_proba(df_test[f_to_use])

df_test['pred'] = preds[:,1]

TRESHOLD = 0.2  # guess, should be tuned with crossval on a subset of train data

d = dict()
for row in df_test.itertuples():
    if row.pred > TRESHOLD:
        try:
            d[row.order_id] += ' ' + str(row.product_id)
        except:
            d[row.order_id] = str(row.product_id)


orders = pd.read_csv(IDIR + 'orders.csv', dtype={
    'order_id': np.int32,
    'user_id': np.int32,
    'eval_set': 'category',
    'order_number': np.int16,
    'order_dow': np.int8,
    'order_hour_of_day': np.int8,
    'days_since_prior_order': np.float32})
test_orders = orders[orders.eval_set == 'test']
for order in test_orders.order_id:
    if order not in d:
        d[order] = 'None'

sub = pd.DataFrame.from_dict(d, orient='index')

sub.reset_index(inplace=True)
sub.columns = ['order_id', 'products']
sub.to_csv('../result/sub_LR.csv', index=False)