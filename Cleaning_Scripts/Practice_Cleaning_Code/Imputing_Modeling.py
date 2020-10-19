#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 13:42:14 2020

@author: aidanosullivan
"""
#imports:
import seaborn as sns
import sklearn as skl
import pandas as pd
import os 
import numpy as np
import math
from collections import Counter

#for auto impute
from autoimpute.imputations import MultipleImputer
import autoimpute as autoimp
from autoimpute.visuals import plot_md_percent
from autoimpute.visuals import plot_md_locations

#for feature selection
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

#for modeling
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
import xgboost as xgb

#establish directory
os.chdir("/Users/aidanosullivan/Desktop/UCLA_Extension/Capstone_Project/Data")


#import data that is clean but has NaN values
data = pd.read_csv("movies_data")


data.columns
#AUTOIMPUTE 

#Now, it is time to impute the data.


#I will follow a 4 step approach to missing data - 
    #1: Assess the extent of the missing value problem with discriptive and visual measures
    #2: Examine factors related to missingness
    #3: Try several imputation methods to determine which is best
    #4: Measure the impact of imputation to the fit, stability, bias, and variance 
    
    

#Step 1: 

#here we see that runtime, status, release_year, and release_month all have missing values. 
data.isna().sum()

# amount of missing data before imputation
print_header("Amount of data missing before imputation takes place")
pd.DataFrame(data.isnull().sum(), columns=["records missing"]).T

autoimp.visuals.plot_md_percent(data.iloc[:,[1,7,8,12,13]])
autoimp.visuals.plot_md_locations(data.iloc[:,[1,7,8,12,13]])

sns.distplot(data['rating'])
sns.distplot(np.log1p(data['revenue']))
sum(data['revenue'] == 0)
sns.distplot(data['popularity'])
sns.distplot(data['vote_count'])

sns.distplot(data['runtime'])
data[data['runtime'] > 1200]


sns.distplot(data['release_year'])
sns.distplot(data['release_month'])

data.status.unique()
data.status.isna().sum()
sum(data['status'] == "In Production")
sum(data['status'] == "Post Production")
sum(data['status'] == "Rumored")
sns.catplot(x = 'status', kind = 'count', palette="ch:.25", data = data)

from seaborn import countplot
countplot(data = data, x = 'status')




#Step 2:

imp = MultipleImputer(
    n = 3
    )    

res = imp.fit_transform(data)
print(res)
res.shape
data.shape

from autoimpute.imputations import SingleImputer
single = SingleImputer(
    strategy = {'status' : "categorical", 'release_year': "median", 'runtime': 'norm', 'release_month': 'random'}
    )
data_imputed_once = single.fit_transform(data)
data_imputed_once.isna().sum()
data_imputed_once.release_month.unique()
data_imputed_once['release_month'] 

triple = MultipleImputer(
    n = 3
    )   

sns.scatterplot(x = 'release_year', y = 'runtime', data = data_imputed_once)
sns.catplot(x = 'release_year', y = 'rating', kind = 'box', data = data_imputed_once)

data = data_imputed_once

#Let's start modeling for god's sake

#get dummies
x = pd.get_dummies(data, columns= ['status'], drop_first = True)
y  = data['rating']

x.drop(['rating', 'imdbId', 'id'], axis = 1, inplace = True)

#split into training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

#find higly correlated variables
correlated_features = set()
correlation_matrix = x.corr()

for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i,j]) > 0.8:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)
print(correlated_features)          

x_train.drop(labels = correlated_features, axis = 1, inplace = True)  
x_test.drop(labels = correlated_features, axis = 1, inplace = True)

#scale the variables

scalar = StandardScaler()
x_train = scalar.fit_transform(x_train)
x_test = scalar.transform(x_test)


#feature selection
rfecv = RFECV(estimator = RandomForestRegressor(), cv = 5, step = 1)
rfecv = rfecv.fit(x_train, y_train.values.ravel())
f = rfecv.get_support(1)

len(f) 
print(f)

parameters = {
              "n_estimators" : [25,50,100, 150, 500],
              "max_depth" : [1,2,4,5,7,9,10,15],
              "min_samples_split" : [2,5],
              "min_samples_leaf" : [1,2,5]
    } 

clf = GridSearchCV(RandomForestRegressor(), parameters, cv = 3, scoring= 'explained_variance')
clf.fit(x_train, y_train.values.ravel())
print(clf.best_params_)
print(clf.score(x_train, y_train.values.ravel()))


#let's try random forest regressor
Rand = RandomForestRegressor(max_depth = 10, min_samples_leaf=5, min_samples_split=5, n_estimators = 500, max_features=int(math.sqrt(len(g))))
rfecv2 = RFECV(estimator = Rand, cv = 3, step = 0.02)
rfecv2 = rfecv2.fit(x_train, y_train.values.ravel())
g = rfecv2.get_support(1)
len(g)
print(g)

#subset x_train1 by the good predictors
x_train1 = pd.DataFrame(x_train)
x_train1 = x_train1[x_train1.columns[g]]

Rand.fit(x_train1, y_train.values.ravel())
y_train_predict = Rand.predict(x_train1)


mean_squared_error(y_train, y_train_predict)
r2_score(y_train, y_train_predict)


x_test1 = pd.DataFrame(x_test)
x_test1 = x_test1[x_test1.columns[g]]
y_test_predict = Rand.predict(x_test1)

mean_squared_error(y_test, y_test_predict)
r2_score(y_test, y_test_predict)


#let's try linear regression, just to see what happens
model = LinearRegression()
model.fit(x_train, y_train)
r_sq = model.score(x_train, y_train)
print(r_sq)

y_pred_lin = model.predict(x_test)
mean_squared_error(y_pred_lin, y_test)

def round_of_rating(number):
    return round(number * 2) / 2


prac = pd.DataFrame(y_test_predict)
rounded = round_of_rating(prac)
test = pd.DataFrame(y_test)
test_rounded = round_of_rating(test)
pd.qcut(rounded[0], 4)
prac2 = pd.Categorical(rounded[0], categories = test_rounded['rating'].unique())
test_cat = pd.Categorical(test_rounded['rating'], categories = test_rounded['rating'].unique())
prac2.shape
rounded
rounded[0].unique()
test_rounded = pd.DataFrame(test_rounded)
Counter(test_rounded['rating'])
rounded[0].dtypes
test_rounded['rating'].unique()
test_rounded
prac2
prac2
pd.crosstab(prac2, test_cat)
cm = metrics.confusion_matrix(test_cat, prac2)
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
test_rounded['rating_cat'] = 0
test
for i in range(len(test_rounded['rating'])):
    print(i)
    if test_rounded.rating.iloc[i] == 0.5:
        test_rounded['rating_cat'] = "Half Star"
    elif test_rounded.rating.iloc[i] == 1.0:
        test_rounded['rating_cat'] = "One Star"
    elif test_rounded.rating.iloc[i] == 1.5:
        test_rounded['rating_cat'] = "One & Half Stars"
    elif test_rounded.rating.iloc[i] == 2.0:
        test_rounded['rating_cat'] = "Two Stars"
    elif test_rounded.rating.iloc[i] == 2.5:
        test_rounded['rating_cat'] = "Two & Half Stars"
    elif test_rounded.rating.iloc[i] == 3.0:
        test_rounded['rating_cat'] = "Three Stars"
    elif test_rounded.rating.iloc[i] == 3.5:
        test_rounded['rating_cat'] = "Three & Half Stars"
    elif test_rounded.rating.iloc[i] == 4.0:
        test_rounded['rating_cat'] = "Four Stars"
    elif test_rounded.rating.iloc[i] == 4.5:
        test_rounded['rating_cat'] = "Four & Half Stars"
    else: # test_rating['rating'][i] = 5:
        test_rounded['rating_cat'] = "Five Stars"
print(test_rounded)
test_rounded.rating.iloc[0] == 3.5

#let's try xgboost

model_xgb = xgb.XGBRegressor(random_state = 42, learning_rate = 0.01)
model_xgb.fit(x_train, y_train)
y_test_xgb = model_xgb.predict(x_test)

model_xgb.score(np.array(y_test_xgb).reshape((-1, 1)), y_test_xgb)
mean_squared_error(y_test, y_test_xgb)
y_test.columns
y_test_xgb
np.array(y_test_xgb)

#lets try polynomial regression
x_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(x_train)
x_2 = PolynomialFeatures(degree=2, include_bias=False).fit_transform(x_test)
model_ply = LinearRegression().fit(x_, y_train)
model_ply.score(x_, y_train)


y_test_ply = model_ply.predict(x_2)
model_ply.score(x_2, y_test)
mean_squared_error(y_test, y_test_ply)
