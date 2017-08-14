import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
pd.set_option('display.width', 320)
#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#



#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#

# features = pd.read_csv('../result/features.csv')
# data = features.iloc[:,4:]
#
# min_max_scaler = preprocessing.MinMaxScaler()
# data = min_max_scaler.fit_transform(data)
#
# labels = features.iloc[:,2]
# logreg = LogisticRegression(C=0.2)
# logreg.fit(data, labels)
# y_pro = logreg.predict_proba(test_data)