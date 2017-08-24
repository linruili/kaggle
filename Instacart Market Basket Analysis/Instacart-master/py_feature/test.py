import pandas as pd
import numpy as np
from itertools import product

pd.set_option('display.width', 320)

tmp_df_train = pd.read_csv('../result/tmp_df_train.csv')
print(tmp_df_train.isnull().values.any())