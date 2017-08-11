import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
pd.set_option('display.width', 320)

#loadTrainData():
index = 761146
features = pd.read_csv('../result/features.csv')
data = features.iloc[:,4:]
labels = features.iloc[:,2]
# train_data, validate_data, train_labels, validate_labels = train_test_split(data, labels, train_size=0.7, random_state=0)
train_data = data.iloc[:index,:]
validate_data = data.iloc[index:,:]
train_labels = labels.iloc[:index]
validate_labels = labels.iloc[index:]

logreg = LogisticRegression()
logreg.fit(train_data, train_labels)
score = logreg.score(validate_data, validate_labels)
print('score = ', score)
y = logreg.predict(validate_data)
predict_features = features.iloc[index:, ]
predict_features['y'] = y

#统计F1 score
predict_features = predict_features[predict_features['y']==1]
predict_num = predict_features.groupby('user_id')['y'].count().reset_index().rename(columns={'y':'predict_num'})

predict_features = predict_features[predict_features['reordered_in_train']==1]
TP = predict_features.groupby('user_id')['y'].count().reset_index().rename(columns={'y':'TP'})

train = pd.read_csv('../result/train.csv')
true_num = train.groupby('user_id')['reordered_in_train'].count().reset_index().rename(columns={'reordered_in_train':'true_num'})
f1 = pd.merge(predict_num, TP)
f1 = pd.merge(f1, true_num)
f1['f1'] = 1/(f1['predict_num']/f1['TP'] + f1['true_num']/f1['TP'])
mean_f1 = f1['f1'].sum()/len(f1['f1'])
print(mean_f1)


