# -*-coding:utf-8-*-
# @Created at: 2019-07-02 17:05
# @Author: Wayne


import pandas as pd
import numpy as np

train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')
combine = [train_df, test_df]


print(train_df.columns.values)
print('*'*80)
print(train_df.head())

print('*'*80)
print(train_df.info())
print('*'*80)
print(test_df.info())

print('*'*80)
# the distribution of numerical feature values across the samples
print(train_df.describe())

print('*'*80)
# the distribution of categorical features
print(train_df.describe(include=[np.object]))



