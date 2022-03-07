# %% read data
#from typing import Collection
import pandas as pd

train = pd.read_csv("titanic/train.csv")
test = pd.read_csv("titanic/test.csv")


# %% checkout out first few rows
train.head()


# %% checkout out dataframe info
train.info()


# %% describe the dataframe
train.describe(include="all")


# %% visualize the dataset, starting with the Survied distribution
import seaborn as sns

sns.countplot(x="Survived", data=train)


# %% Survived w.r.t Pclass / Sex / Embarked ?
sns.countplot(x="Survived",hue="Pclass", data=train)
# %%Survived w.r.t Sex 
sns.countplot(x="Survived",hue="Sex", data=train)
# %% Survived w.r.t Embarked 
sns.countplot(x="Survived",hue="Embarked", data=train)

# %% Age distribution ?
sns.distplot(train["Age"])

# %% Survived w.r.t Age distribution ?

sns.displot(data=train, x= "Age", hue="Survived")

#%%
sns.distplot(train[train["Survived"]==1]["Age"],label="survived")
sns.distplot(train[train["Survived"]==0]["Age"],label="Died")
import matplotlib.pyplot as plt
plt.legend()

# %% SibSp / Parch distribution ?
sns.distplot(train["SibSp"])
# %%
sns.distplot(train["Parch"])


# %% Survived w.r.t SibSp / Parch  ?
sns.countplot(x="Survived",hue="SibSp", data=train)
# %% 
sns.countplot(x="Survived",hue="Parch", data=train)


# %% Dummy Classifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score

def evaluate(clf, x, y):
    pred = clf.predict(x)
    result = f1_score(y, pred)
    return f"F1 score: {result:.3f}"


dummy_clf = DummyClassifier(random_state=2020)

dummy_selected_columns = ["Pclass"]
dummy_train_x = train[dummy_selected_columns]
dummy_train_y = train["Survived"]

dummy_clf.fit(dummy_train_x, dummy_train_y)
print("Training Set Performance")
print(evaluate(dummy_clf, dummy_train_x, dummy_train_y))

truth = pd.read_csv("truth_titanic.csv")
dummy_test_x = test[dummy_selected_columns]
dummy_test_y = truth["Survived"]

print("Test Set Performance")
print(evaluate(dummy_clf, dummy_test_x, dummy_test_y))

print("Can you do better than a dummy classifier?")


# %% Your solution to this classification problem

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

#add the following
imp = SimpleImputer()
enc = OneHotEncoder(handle_unknown="ignore")
ct = ColumnTransformer(
    [
        ("new_age", imp , ["Age"]),
        ("new_sex",enc,["Sex"]),
    ],
    remainder="passthrough"
)

#clf = KNeighborsClassifier()
clf = MLPClassifier()

selected_columns = ["Pclass", "Age","Sex"]
train_x = train[selected_columns]
train_y = train["Survived"]

train_x = ct.fit_transform(train_x) #add this line

clf.fit(train_x, train_y)
print("Training Set Performance")
print(evaluate(clf, train_x, train_y))

truth = pd.read_csv("truth_titanic.csv")
test_x = test[selected_columns]
test_y = truth["Survived"]

test_x = ct.fit_transform(test_x)#add this line

print("Test Set Performance")
print(evaluate(clf, test_x, test_y))
# %%
# %% Your solution to this classification problem
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

#add the following
imp = SimpleImputer()
enc = OneHotEncoder(handle_unknown="ignore")
ct = ColumnTransformer(
    [
        ("new_age", imp , ["Age"]),
        ("new_sex",enc,["Sex"]),
    ],
    remainder="passthrough"
)

#clf = KNeighborsClassifier()
clf = MLPClassifier()

selected_columns = ["Pclass", "Age","Sex"]
train_x = train[selected_columns]
train_y = train["Survived"]

train_x = ct.fit_transform(train_x) #add this line

clf.fit(train_x, train_y)
print("Training Set Performance")
print(evaluate(clf, train_x, train_y))

truth = pd.read_csv("truth_titanic.csv")
test_x = test[selected_columns]
test_y = truth["Survived"]

test_x = ct.fit_transform(test_x)#add this line

print("Test Set Performance")
print(evaluate(clf, test_x, test_y))

# %%
("kfjfkj")