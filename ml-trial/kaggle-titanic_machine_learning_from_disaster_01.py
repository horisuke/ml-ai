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
# sns.factorplot('Embarked', 'Survived', data=titanic_df, size=4, aspect=3)

fig, (axis1, axis2, axis3) = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
# Embarked : plot -- Count Bar of Embarked
sns.countplot(x='Embarked', data=titanic_df, ax=axis1)

# Embarked : plot -- Count Bar of Survived by Embarked
sns.countplot(x='Survived', hue="Embarked", data=titanic_df, order=[1, 0], ax=axis2)

# Embarked : plot -- average of Survived in each Embarked
embark_perc = titanic_df[["Embarked", "Survived"]].groupby(['Embarked'], as_index=False).mean()
sns.barplot(x='Embarked', y='Survived', data=embark_perc, order=['S', 'C', 'Q'], ax=axis3)

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
embark_dummies_test = pd.get_dummies(test_df['Embarked'])

# Embarked : Drop unnecessary 'S' columns for test data(C:10, Q:01, S:00)
embark_dummies_test.drop(['S'], axis=1, inplace=True)

# Embarked : Join dummy data(C, Q) to each train data & test data
titanic_df = titanic_df.join(embark_dummies_titanic)
test_df = test_df.join(embark_dummies_test)

# Embarked : Drop unnecessary 'Embarked' column in train data & test data
titanic_df.drop(['Embarked'], axis=1, inplace=True)
test_df.drop(['Embarked'], axis=1, inplace=True)
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
rand_1 = np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic,
                           count_nan_age_titanic)
rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, count_nan_age_test)
print("rand_1:" + str(rand_1))
print("rand_2:" + str(rand_2))

# Age : create fig & plot
fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(15, 4))
axis1.set_title('Original Age values - Titanic')
axis2.set_title('New Age values - Titanic')

# Age : Draw the histogram of "Age"
titanic_df['Age'].dropna().astype(int).hist(bins=70, ax=axis1)

# Age : Insert generated random values to the missing values of "Age"
titanic_df['Age'][np.isnan(titanic_df['Age'])] = rand_1
test_df["Age"][np.isnan(test_df["Age"])] = rand_2
titanic_df['Age'] = titanic_df['Age'].astype(int)
test_df['Age'] = test_df['Age'].astype(int)

# Age : Draw the histogram of "Age" after adding random values in missing values
titanic_df['Age'].dropna().astype(int).hist(bins=70, ax=axis2)

# Age : display the histogram of "Age"
plt.show()

# Age : peaks for survived/not survived passengers by their age
facet = sns.FacetGrid(titanic_df, hue="Survived", aspect=4)
facet.map(sns.kdeplot, 'Age', shade=True)
facet.set(xlim=(0, titanic_df['Age'].max()))
facet.add_legend()
plt.show()

# Age : average survived passengers by age
fig, axis1 = plt.subplots(1, 1, figsize=(18, 4))
average_age = titanic_df[["Age", "Survived"]].groupby(['Age'], as_index=False).mean()
sns.barplot(x='Age', y='Survived', data=average_age)
plt.show()
print("-----------------average_age.head()")
print(average_age.head())
print("-----------------titanic_df.head()")
print(titanic_df.head())

# Cabin : Drop Cabin column
# It has a lot of NaN values, so it won't cause a remarkable impact on prediction.
titanic_df.drop("Cabin", axis=1, inplace=True)
test_df.drop("Cabin", axis=1, inplace=True)
print("-----------------titanic_df.head()")
print(titanic_df.head())

# Family : Create new column as 'Family'
# Instead of having two columns Parch & SibSp,
# we can have only one column represent if the passenger had any family member aboard or not,
# Meaning, if having any family member(whether parent, brother, ...etc) will increase chances of Survival or not.
titanic_df['Family'] = titanic_df["Parch"] + titanic_df["SibSp"]
titanic_df['Family'].loc[titanic_df['Family'] > 0] = 1
titanic_df['Family'].loc[titanic_df['Family'] == 0] = 0
print("-----------------titanic_df.head()")
print(titanic_df.head())
test_df['Family'] = test_df["Parch"] + test_df["SibSp"]
test_df['Family'].loc[test_df['Family'] > 0] = 1
test_df['Family'].loc[test_df['Family'] == 0] = 0

# Family : Drop Parch & SibSp columns
titanic_df = titanic_df.drop(['SibSp', 'Parch'], axis=1)
test_df = test_df.drop(['SibSp', 'Parch'], axis=1)
print("-----------------titanic_df.head()")
print(titanic_df.head())

# Family : plot
fig, (axis1, axis2) = plt.subplots(1, 2, sharex=True, figsize=(10, 5))

# Family : count of Survived / Not survived with family or alone.
sns.countplot(x='Family', data=titanic_df, order=[1, 0], ax=axis1)
axis1.set_xticklabels(["With Family", "Alone"], rotation=90)

# Family : average of survived those who had/did't any family.
family_perc = titanic_df[["Family", "Survived"]].groupby(['Family'], as_index=False).mean()
sns.barplot(x='Family', y='Survived', data=family_perc, order=[1, 0], ax=axis2)
axis2.set_xticklabels(["With Family", "Alone"], rotation=270)
plt.show()
print("-----------------titanic_df.head()")
print(titanic_df.head())


# Sex :
# Children (age < 16) on aboard seem to have a high chances for Survival.
# So, classify passengers as males, females, and child

# Sex : define get_person function for return child, male, or female
def get_person(passenger):
    age, sex = passenger
    return 'child' if age < 16 else sex

# Sex : create Person column using values of Age / Sex
titanic_df['Person'] = titanic_df[['Age', 'Sex']].apply(get_person, axis=1)
test_df['Person'] = test_df[['Age', 'Sex']].apply(get_person, axis=1)

# Sex : drop Sex column insted of Person column
titanic_df.drop(['Sex'], axis=1, inplace=True)
test_df.drop(['Sex'], axis=1, inplace=True)
print("-----------------titanic_df.head()")
print(titanic_df.head())

# Sex : change the categorical variables(child, female, male) of Person to dummy variables for train data
person_dummies_titanic = pd.get_dummies(titanic_df['Person'])
person_dummies_titanic.columns = ['Child', 'Female', 'Male']
person_dummies_titanic.drop(['Male'], axis=1, inplace=True)

# Sex : change the categorical variables(child, female, male) of Person to dummy variables for test data
person_dummies_test = pd.get_dummies(test_df['Person'])
person_dummies_test.columns = ['Child', 'Female', 'Male']
person_dummies_test.drop(['Male'], axis=1, inplace=True)

# Sex : Join dummy data to each train data & test data
titanic_df = titanic_df.join(person_dummies_titanic)
test_df = test_df.join(person_dummies_test)
print("-----------------titanic_df.head()")
print(titanic_df.head())

# Sex : plot
fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(10, 5))

# Sex : count of Survived / Not survived with male, female, child.
sns.countplot(x='Person', data=titanic_df, ax=axis1)

# Sex : average of survived for each person(male, female, or child)
person_perc = titanic_df[["Person", "Survived"]].groupby(['Person'], as_index=False).mean()
sns.barplot(x='Person', y='Survived', data=person_perc, order=['male', 'female', 'child'], ax=axis2)
plt.show()

# Sex : drop Person column
titanic_df.drop(['Person'], axis=1, inplace=True)
test_df.drop(['Person'], axis=1, inplace=True)
print("-----------------titanic_df.head()")
print(titanic_df.head())


# Pclass : plot
sns.catplot('Pclass', 'Survived', order=[1,2,3], data=titanic_df, kind='point')
plt.show()

# Pclass : change the categorical variables(1, 2, 3) of Pclass to dummy variables for train data
pclass_dummies_titanic = pd.get_dummies(titanic_df['Pclass'])
pclass_dummies_titanic.columns = ['Class1', 'Class2', 'Class3']
pclass_dummies_titanic.drop(['Class3'], axis=1, inplace=True)

# Pclass : change the categorical variables(1, 2, 3) of Pclass to dummy variables for test data
pclass_dummies_test = pd.get_dummies(test_df['Pclass'])
pclass_dummies_test.columns = ['Class1', 'Class2', 'Class3']
pclass_dummies_test.drop(['Class3'], axis=1, inplace=True)

# Pclass : drop Person column
titanic_df.drop(['Pclass'], axis=1, inplace=True)
test_df.drop(['Pclass'], axis=1, inplace=True)
print("-----------------titanic_df.head()")
print(titanic_df.head())

# Pclass : Join dummy data to each train data & test data
titanic_df = titanic_df.join(pclass_dummies_titanic)
test_df = test_df.join(pclass_dummies_test)
print("-----------------titanic_df.head()")
print(titanic_df.head())
print("-----------------titanic_df.describe()")
print(titanic_df.describe())
print("-----------------titanic_df.corr()")
print(titanic_df.corr())

# Define train & test data
X_train = titanic_df.drop("Survived", axis=1)
Y_train = titanic_df["Survived"]
X_test = test_df.drop("PassengerId", axis=1).copy()

# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
print("-----------------logreg.score(X_train, Y_train)")
print(logreg.score(X_train, Y_train))
print("-----------------logreg.coef_")
print(logreg.coef_)
print(logreg.intercept_)
# ---------------

# Support Vector Machines
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
print("-----------------svc.score(X_train, Y_train)")
print(svc.score(X_train, Y_train))

# Random Forests
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
print("-----------------random_forest.score(X_train, Y_train)")
print(random_forest.score(X_train, Y_train))

# k-nearest neighbor algorithm
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
print("-----------------knn.score(X_train, Y_train)")
print(knn.score(X_train, Y_train))

# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
print("-----------------gaussian.score(X_train, Y_train)")
print(gaussian.score(X_train, Y_train))



