# Competition     : Titanic: Machine Learning from Disaster
# Competition URL : https://www.kaggle.com/c/titanic
# Reference EDA   : https://www.kaggle.com/omarelgabry/a-journey-through-titanic

# Import

# pandas
import pandas as pd
from pandas import Series, DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# Data summary

# get titanic & test csv file as DataFrame
pd.set_option('display.max_columns', None)
titanic_df = pd.read_csv("data/titanic/train.csv")
test_df = pd.read_csv("data/titanic/test.csv")

# preview the train & test data
print("-----------------titanic_df.head()")
print(titanic_df.head())
print("-----------------titanic_df.info()")
print(titanic_df.info())
print("-----------------test_df.info()")
print(test_df.info())

# check the all column names
print("-----------------titanic_df.columns.values")
print(titanic_df.columns.values)

# check whether there are the missing values or not in each column
print("-----------------titanic_df.isnull().any(axis=0)")
print(titanic_df.isnull().any(axis=0))

# count the number of missing values in each column
print("-----------------titanic_df.isnull().sum(axis=0)")
print(titanic_df.isnull().sum(axis=0))

# check the count of values in columns including missing data
print("-----------------titanic_df[Embarked].value_counts()")
print(titanic_df["Embarked"].value_counts())
print("-----------------titanic_df[Age].value_counts()")
print(titanic_df["Age"].value_counts())
print("-----------------titanic_df[Cabin].value_counts()")
print(titanic_df["Cabin"].value_counts())


# Preprocessing

# Drop unnecessary columns, these columns won't be useful in analysis and prediction.
titanic_df = titanic_df.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
test_df = test_df.drop(['Name', 'Ticket'], axis=1)

# Embarked : fill the missing values with the occurred value, which is "S" only in titanic_df
titanic_df["Embarked"] = titanic_df["Embarked"].fillna("S")

# Embarked : plot -- Embarked * Survived
sns.catplot('Embarked', 'Survived', data=titanic_df, height=4, aspect=3, kind='point')
#sns.factorplot('Embarked', 'Survived', data=titanic_df, size=4, aspect=3)

fig, (axis1, axis2, axis3) = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
# Embarked : plot -- Count Bar of Embarked
sns.countplot(x='Embarked', data=titanic_df, ax=axis1)

# Embarked : plot -- Count Bar of Survived by Embarked
sns.countplot(x='Survived', hue="Embarked", data=titanic_df, order=[1,0], ax=axis2)

# Embarked : plot -- average of Survived in each Embarked
embark_perc = titanic_df[["Embarked", "Survived"]].groupby(['Embarked'], as_index=False).mean()
sns.barplot(x='Embarked', y='Survived', data=embark_perc, order=['S','C','Q'], ax=axis3)

# Embarked : show plot
print("-----------------plt.show()")
plt.show()

# Embarked : change the categorical variables(S, C, Q) of Embarked to dummy variables for train data
embark_dummies_titanic = pd.get_dummies(titanic_df['Embarked'])
print("-----------------embark_dummies_titanic")
print(embark_dummies_titanic)

# Embarked : Drop unnecessary 'S' columns for train data(C:10, Q:01, S:00)
embark_dummies_titanic.drop(['S'], axis=1, inplace=True)

# Embarked : change the categorical variables(S, C, Q) of Embarked to dummy variables for test data
embark_dummies_test  = pd.get_dummies(test_df['Embarked'])

# Embarked : Drop unnecessary 'S' columns for test data(C:10, Q:01, S:00)
embark_dummies_test.drop(['S'], axis=1, inplace=True)

# Embarked : Join dummy data(C, Q) to each train data & test data
titanic_df = titanic_df.join(embark_dummies_titanic)
test_df = test_df.join(embark_dummies_test)

# Embarked : Drop unnecessary 'Embarked' column in train data & test data
titanic_df.drop(['Embarked'], axis=1,inplace=True)
test_df.drop(['Embarked'], axis=1,inplace=True)
print("-----------------titanic_df.head()")
print(titanic_df.head())


# Fare : fill the missing values with the median value in test_df
test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)

# Fare : convert all values from float to int in train data & test data
titanic_df['Fare'] = titanic_df['Fare'].astype(int)
test_df['Fare'] = test_df['Fare'].astype(int)

# Fare : get fare for survived & didn't survive passengers
fare_not_survived = titanic_df["Fare"][titanic_df["Survived"] == 0]
fare_survived = titanic_df["Fare"][titanic_df["Survived"] == 1]
print("-----------------fare_not_survived")
print(fare_not_survived)

# Fare : get average and std for fare of survived/not survived passengers
average_fare = DataFrame([fare_not_survived.mean(), fare_survived.mean()])
print("-----------------average_fare")
print(average_fare)
std_fare = DataFrame([fare_not_survived.std(), fare_survived.std()])
print("-----------------std_fare")
print(std_fare)

# Fare : plot
sns.distplot(titanic_df['Fare'][titanic_df["Fare"] <= 50], hist=True, kde=False, rug=False, bins=10)
plt.show()


# Age : get average, std and number of NaN for age in titanic_df
average_age_titanic = titanic_df["Age"].mean()
std_age_titanic = titanic_df["Age"].std()
count_nan_age_titanic = titanic_df["Age"].isnull().sum()
print("average_age_titanic:" + str(average_age_titanic))
print("std_age_titanic:" + str(std_age_titanic))
print("count_nan_age_titanic:" + str(count_nan_age_titanic))
print("-----------------")

# Age : get average, std and number of NaN for age in test_df
average_age_test = test_df["Age"].mean()
std_age_test = test_df["Age"].std()
count_nan_age_test = test_df["Age"].isnull().sum()
print("average_age_test:" + str(average_age_test))
print("std_age_test:" + str(std_age_test))
print("count_nan_age_test:" + str(count_nan_age_test))

# Age : generate random numbers between (mean - std) & (mean + std)
rand_1 = np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic, count_nan_age_titanic)
rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, count_nan_age_test)
print("rand_1:" + str(rand_1))
print("rand_2:" + str(rand_2))

# Age : plot
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))
axis1.set_title('Original Age values - Titanic')
axis2.set_title('New Age values - Titanic')

titanic_df['Age'].dropna().astype(int).hist(bins=70, ax=axis1)

titanic_df['Age'][np.isnan(titanic_df['Age'])] = rand_1
test_df["Age"][np.isnan(test_df["Age"])] = rand_2

titanic_df['Age'] = titanic_df['Age'].astype(int)
test_df['Age']    = test_df['Age'].astype(int)

titanic_df['Age'].dropna().astype(int).hist(bins=70, ax=axis2)

plt.show()

print("-----------------titanic_df.head()")
print(titanic_df.head())




