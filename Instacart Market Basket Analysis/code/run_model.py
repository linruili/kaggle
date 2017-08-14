import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
pd.set_option('display.width', 320)

#loadTrainData():
index = 416446
features = pd.read_csv('../result/features.csv')
data = features.iloc[:,4:]
data = data.iloc[:,:8]

min_max_scaler = preprocessing.MinMaxScaler()
data = min_max_scaler.fit_transform(data)

labels = features.iloc[:,2]
# train_data, validate_data, train_labels, validate_labels = train_test_split(data, labels, train_size=0.7, random_state=0)
train_data = data[index:,:]
validate_data = data[:index,:]
train_labels = labels.iloc[index:]
validate_labels = labels.iloc[:index]

mean_f1 = []
for i in range(1,10):
    logreg = LogisticRegression(C=0.12+0.01*i)
    logreg.fit(train_data, train_labels)
    score = logreg.score(validate_data, validate_labels)
    print('score = ', score)
    y_pro = logreg.predict_proba(validate_data)

    y = preprocessing.binarize(y_pro, 0.2)[:,1]
    predict_features = features.iloc[:index, :]
    predict_features['y'] = y

    #统计F1 score
    predict_features = predict_features[predict_features['y']==1]
    predict_num = predict_features.groupby('user_id')['y'].count().reset_index().rename(columns={'y':'predict_num'})

    predict_features = predict_features[predict_features['reordered_in_train']==1]
    TP = predict_features.groupby('user_id')['y'].count().reset_index().rename(columns={'y':'TP'})

    train = pd.read_csv('../result/train.csv')
    true_num = train.groupby('user_id')['add_to_cart_order'].max().reset_index().rename(columns={'add_to_cart_order':'true_num'})
    f1 = pd.merge(predict_num, TP)

    f1 = pd.merge(f1, true_num)
    f1['f1'] = 2/(f1['predict_num']/f1['TP'] + f1['true_num']/f1['TP'])
    mean_f1.append(f1['f1'].sum()/len(f1['f1']))
print(mean_f1)
plt.figure()
plt.plot(np.arange(1,10)*0.01+0.13, mean_f1)
plt.show()


