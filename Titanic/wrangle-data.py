# -*-coding:utf-8-*-
# @Created at: 2019-07-04 15:50
# @Author: Wayne

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')

print('Before:', train_df.shape, test_df.shape)

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)

combine = [train_df, test_df]

print('After:', combine[0].shape, combine[1].shape)

# Create new feature extracting from existing
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

print(pd.crosstab(train_df['Title'], train_df['Sex']))

# Replace many titles with a more common name or classify them as rare.
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(\
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

print(train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

# Convert the categorical titles to ordinal.
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

print(train_df.head())

# Drop the Name feature from training and testing data
# Drop the PassengerId in the training data.
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
print(train_df.shape, test_df.shape)

# Convert a categorical feature (Sex---0:male, 1:female)
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0}).astype(int)

print(train_df.head())

# Completing a numerical  continuous feature
grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
# plt.show()

guess_ages = np.zeros((2, 3))
for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                               (dataset['Pclass'] == j + 1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5

    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j + 1), \
                        'Age'] = guess_ages[i, j]

    dataset['Age'] = dataset['Age'].astype(int)

print(train_df.head())

# Create Age bands and determine correlations with Survived.
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
print(train_df[['AgeBand', 'Survived']].groupby \
          (['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True))

# Replace Age with ordinals based on these bands.
for dataset in combine:
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age']
print(train_df.head())

# Drop AgeBand feature.
train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
print(train_df.head())

# Create new feature combining existing features.
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

print(train_df[['FamilySize', 'Survived']].groupby \
          (['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False))

# Create another feature called IsAlone.
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

print(train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())

# Let us drop Parch, SibSp and FamilySize features in flavor of IsAlone.
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

print(train_df.head())

# Create an artificial feature combining Pclass and Age
for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

print(train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10))


# Completing a categorical feature.
freq_port = train_df.Embarked.dropna().mode()[0]

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

print(train_df[['Embarked', 'Survived']].groupby\
    (['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False))

# Convert categorical feature to numerical.
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

print(train_df.head())

# Quick completing and converting a numerical feature.
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
print(test_df.head())

# Create feature FareBand.
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
print(train_df[['FareBand', 'Survived']].groupby\
    (['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True))

# Convert the Fare feature to ordinal values based on the FareBand.
for dataset in combine:
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]

print(train_df.head(10))
print(test_df.head(10))

# Save training and testing DataSet.
train_df.to_csv('./data/train_mdf.csv', index=False, sep=',')
test_df.to_csv('./data/test_mdf.csv', index=False, sep=',')





