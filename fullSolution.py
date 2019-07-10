
# -*- coding: utf-8 -*-
import os 
import pandas as pd
import numpy as np
import math
import category_encoders as ce
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc
import scipy.stats as stats
import statsmodels.api as sm
from random import randint, random
from sklearn import svm 
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt


#os.chdir('C:/Users/Lendflo-Przemek/Documents/Tmobile')

from Helper import Helper 

helper = Helper()
 
# First model
train = pd.read_csv("TrainDataset.csv", delimiter ='\t')
train = train.set_index('CustomerID')

train['Churn'] = pd.factorize(train['Churn'])[0]  
Y = train[['Churn']]
train = train.drop(['Churn'], axis = 1)

train = helper.deletingObjectColumns(train)
train = train.dropna()

Y = Y.loc[Y.index.intersection(train.index)]
X_train, X_validate, y_train, y_validate = train_test_split(train,Y, test_size = 0.1, random_state = 1)

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0).fit(X_train, y_train)
print("1 MODEL (Plain with dropna and deleting categorical variables ) ACC" , clf.score(X_validate, y_validate))
print("1 MODEL (Plain with dropna and deleting categorical variables) AUC training GBC", roc_auc_score(y_train[0:200], clf.predict(X_train[0:200])))
print("1 MODELAUC (Plain with dropna and deleting categorical variables) validation GBC", roc_auc_score(y_validate, clf.predict(X_validate)))

#Socond model
helper = Helper()
train = pd.read_csv("TrainDataset.csv", delimiter ='\t')
train = train.set_index('CustomerID')

train['Churn'] = pd.factorize(train['Churn'])[0]  
Y = train[['Churn']]


train = helper.deletingObjectColumns(train)

train = helper.cleaningNA(train)

train = helper.scaling(train)

train = train.drop(['Churn'], axis = 1)

Y = Y.loc[Y.index.intersection(train.index)]
X_train, X_validate, y_train, y_validate = train_test_split(train,Y, test_size = 0.1, random_state = 1)

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0).fit(X_train, y_train)
print("2 MODEL (With scaling and cleanin NA) ACC" , clf.score(X_validate, y_validate))
print("2 MODEL (With scaling and cleanin NA) AUC training GBC", roc_auc_score(y_train[0:200], clf.predict(X_train[0:200])))
print("2 MODELAUC (With scaling and cleanin NA) validation GBC", roc_auc_score(y_validate, clf.predict(X_validate)))

#Third model
helper = Helper()
train = pd.read_csv("TrainDataset.csv", delimiter ='\t')
train = train.set_index('CustomerID')

train['Churn'] = pd.factorize(train['Churn'])[0]  
Y = train[['Churn']]

train['Location'] = train['Location'].apply(lambda x: str(x)[0:3])
train = helper.cleaningNA(train)
train = helper.scaling(train)
train = helper.binning(train)

train = train.drop(['Churn'], axis = 1)

Y = Y.loc[Y.index.intersection(train.index)]
X_train, X_validate, y_train, y_validate = train_test_split(train,Y, test_size = 0.1, random_state = 1)

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0).fit(X_train, y_train)
print("3 MODEL (With modified location, not deleting categorical  and binning) ACC" , clf.score(X_validate, y_validate))
print("3 MODEL (With modified location, not deleting categorical  and binning) AUC training GBC", roc_auc_score(y_train[0:200], clf.predict(X_train[0:200])))
print("3 MODEL (With modified location, not deleting categorical  and binning) AUC validation GBC", roc_auc_score(y_validate, clf.predict(X_validate)))

#Fourth model
helper = Helper()
train = pd.read_csv("TrainDataset.csv", delimiter ='\t')
train = train.set_index('CustomerID')

train['Churn'] = pd.factorize(train['Churn'])[0]  
Y = train[['Churn']]

train['Location'] = train['Location'].apply(lambda x: str(x)[0:3])
train = helper.cleaningNA(train)
train = helper.scaling(train)
train = helper.binning(train)
train = helper.cleaningOutliers(train)

train = train.drop(['Churn'], axis = 1)

Y = Y.loc[Y.index.intersection(train.index)]
X_train, X_validate, y_train, y_validate = train_test_split(train,Y, test_size = 0.1, random_state = 1)

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0).fit(X_train, y_train)
print("4 MODEL (With cleaning outliers) ACC" , clf.score(X_validate, y_validate))
print("4 MODEL (With cleaning outliers) AUC training GBC", roc_auc_score(y_train[0:200], clf.predict(X_train[0:200])))
print("4 MODELAUC (With cleaning outliers) validation GBC", roc_auc_score(y_validate, clf.predict(X_validate)))


#Fifth model
train = pd.read_csv("TrainDataset.csv", delimiter ='\t')
train = train.set_index('CustomerID')

train['Churn'] = pd.factorize(train['Churn'])[0] 
Y = train[['Churn']]

train['Location'] = train['Location'].apply(lambda x: str(x)[0:3])
train = helper.cleaningNA(train)
train ['decimalPrice'] = train.apply(lambda x : helper.gettingDecimals(x, 'PhonePrice'), axis = 1 )
train = helper.scaling(train)
train = helper.binning(train)
train = helper.cleaningOutliers(train)


train = train.drop(['Churn'], axis = 1)

Y = Y.loc[Y.index.intersection(train.index)]
X_train, X_validate, y_train, y_validate = train_test_split(train,Y, test_size = 0.1, random_state = 1)

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0).fit(X_train, y_train)
print("5 MODEL (With decimal price extraction) ACC" , clf.score(X_validate, y_validate))
print("5 MODEL (With decimal price extraction) AUC training GBC", roc_auc_score(y_train[0:200], clf.predict(X_train[0:200])))
print("5 MODELAUC (With decimal price extraction) validation GBC", roc_auc_score(y_validate, clf.predict(X_validate)))

#Sixth model
train = pd.read_csv("TrainDataset.csv", delimiter ='\t')

train = train.set_index('CustomerID')

train['Churn'] = pd.factorize(train['Churn'])[0] 

Y = train[['Churn']]

train['Location'] = train['Location'].apply(lambda x: str(x)[0:3])
train['oddOrEven'] = train.apply(lambda x : helper.getLastDigit(x), axis = 1 ) 

train = helper.cleaningNA(train)

train ['decimalPrice'] = train.apply(lambda x : helper.gettingDecimals(x, 'PhonePrice'), axis = 1 )

train = helper.binning(train)
train = helper.factorizingCategorical(train)
train = helper.scaling(train)

train = helper.cleaningOutliers(train)
train = train[train['Churn']!= -1]

train = train.drop(['Churn'], axis = 1)


Y = Y.loc[Y.index.intersection(train.index)]

X_train, X_validate, y_train, y_validate = train_test_split(train,Y, test_size = 0.1, random_state = 1)

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0).fit(X_train, y_train)
print("6 MODEL (Wih odd and even id distinguisher) ACC" , clf.score(X_validate, y_validate))
print("6 MODEL (Wih odd and even id distinguisher) AUC training GBC", roc_auc_score(y_train[0:200], clf.predict(X_train[0:200])))
print("6 MODELAUC (Wih odd and even id distinguisher) validation GBC", roc_auc_score(y_validate, clf.predict(X_validate)))
 

#Model 7th
train = pd.read_csv("TrainDataset.csv", delimiter ='\t')
train = train.set_index('CustomerID')

train['Churn'] = pd.factorize(train['Churn'])[0] 

Y = train[['Churn']]

train['Location'] = train['Location'].apply(lambda x: str(x)[0:3])
train['oddOrEven'] = train.apply(lambda x : helper.getLastDigit(x), axis = 1 ) 

train = helper.cleaningNA(train)

train ['decimalPrice'] = train.apply(lambda x : helper.gettingDecimals(x, 'PhonePrice'), axis = 1 )

train = helper.binning(train)
train = helper.factorizingCategorical(train)
train = helper.scaling(train)
train = helper.cleaningOutliers(train)

train = train.drop(['Churn'], axis = 1)

Y = Y.loc[Y.index.intersection(train.index)]

X_train, X_validate, y_train, y_validate = train_test_split(train,Y, test_size = 0.1, random_state = 1)

svc = svm.SVC(gamma="scale")
modelSVC = svc.fit(X_train,y_train)

print("7 MODEL (With SVM) ACC" , svc.score(X_validate, y_validate))
print("7 MODEL (With SVM)  AUC training SVM", roc_auc_score(y_train[0:200], modelSVC.predict(X_train[0:200])))
print("7 MODELAUC (With SVM) validation SVM", roc_auc_score(y_validate, modelSVC.predict(X_validate)))


#TEST CHECK


train = pd.read_csv("TrainDataset.csv", delimiter ='\t')
test = pd.read_csv("TestDataset.csv", delimiter ='\t')
train = train.append(test)
train = train.set_index('CustomerID')

train['Churn'] = pd.factorize(train['Churn'])[0] 

Y = train[['Churn']]

train['Location'] = train['Location'].apply(lambda x: str(x)[0:3])
train['oddOrEven'] = train.apply(lambda x : helper.getLastDigit(x), axis = 1 ) 

train = helper.cleaningNA(train)

train ['decimalPrice'] = train.apply(lambda x : helper.gettingDecimals(x, 'PhonePrice'), axis = 1 )

train = helper.binning(train)
train = helper.factorizingCategorical(train)
train = helper.scaling(train)

#train = helper.cleaningOutliers(train)
test = train[train['Churn']== -1]
train = train[train['Churn']!= -1]

train = train.drop(['Churn'], axis = 1)
test = test.drop(['Churn'], axis = 1)


Y = Y.loc[Y.index.intersection(train.index)]

X_train, X_validate, y_train, y_validate = train_test_split(train,Y, test_size = 0.1, random_state = 1)

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0).fit(X_train, y_train)

#### ADD test to train
results = pd.read_csv("CorrectDataset.csv", delimiter ='\t')
results = results.set_index('CustomerID')


print("TEST MODEL (Wih odd and even id distinguisher) ACC" , clf.score(test, results))
print("TEST MODELAUC (Wih odd and even id distinguisher) validation GBC", roc_auc_score(results, clf.predict(test)))





