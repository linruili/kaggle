import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
pd.set_option('display.width', 320)
#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#
features = pd.read_csv('../result/features.csv')
features_test = pd.read_csv('../result/features_test.csv')
features_test = features_test.fillna(0.013)

data = features.iloc[:,4:]
test_data = features_test.iloc[:,2:]

min_max_scaler = preprocessing.MinMaxScaler()
data = min_max_scaler.fit_transform(data)

labels = features.iloc[:,2]
logreg = LogisticRegression(C=0.2)
logreg.fit(data, labels)
y_pro = logreg.predict_proba(test_data)

y = preprocessing.binarize(y_pro, 0.2)[:,1]
predict_result = features_test
predict_result['y'] = y
#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#
def f6(arr):
    if(arr['y'].sum()==0):
        return 'None'
    arr = arr[arr['y']==1]
    ans = ""
    for i in arr['product_id'].values:
        ans += (str(i)+" ")
    return ans.rstrip()

predict_result = predict_result[['user_id', 'product_id', 'y']]
# is_none = predict_result.groupby('user_id')['y'].sum().reset_index().rename(columns={'y':'is_none'})
# is_none.loc[is_none['is_none']>0, 'is_none'] = 1
# predict_result = pd.merge(predict_result, is_none)

orders = pd.read_csv('../data/orders.csv')
orders = orders[orders['eval_set']=='test']
orders = orders[['order_id', 'user_id']]
predict_result = pd.merge(orders, predict_result)
result = predict_result.groupby('order_id').apply(f6)
result = pd.DataFrame({result.index.name:result.index.values, 'products':result.values}, index=range(len(result)))

print(result)
result.to_csv('../result/result_2.csv', index=False)