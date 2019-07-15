# -*-coding:utf-8-*-
# @Created at: 2019-07-03 16:06
# @Author: Wayne

# make sense doing so only for features which are categorical.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


train_df = pd.read_csv('./data/train.csv')

# Pclass
print(train_df[['Pclass', 'Survived']].groupby(['Pclass'], \
        as_index=False).mean().sort_values(by='Survived', ascending=False))

# Sex
print(train_df[["Sex", "Survived"]].groupby(['Sex'], \
        as_index=False).mean().sort_values(by='Survived', ascending=False))

# SibSp
print(train_df[["SibSp", "Survived"]].groupby(['SibSp'], \
        as_index=False).mean().sort_values(by='Survived', ascending=False))

# Parch
print(train_df[["Parch", "Survived"]].groupby(['Parch'], \
        as_index=False).mean().sort_values(by='Survived', ascending=False))

# Visualize
# Correlating numerical features: Age
g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)
plt.show()

# Correlating numerical and ordinal features: Age and Pclass
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', height=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=0.5, bins=20)
grid.add_legend()
plt.show()

# Correlating categorical features
grid = sns.FacetGrid(train_df, row='Embarked', height=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()
plt.show()


# Correlating categorical and numerical features
grid = sns.FacetGrid(train_df, row='Embarked', height=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=0.5, ci=None)
grid.add_legend()
plt.show()

