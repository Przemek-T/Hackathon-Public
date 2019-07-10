
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 10:26:49 2019

@author: Przemek
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import category_encoders as ce
from sklearn.preprocessing import StandardScaler, StandardScaler, Imputer
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from random import randint, random
import math
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import Callback

class Helper:
        
    def scaling(self, dataset):
        for column in list(dataset.select_dtypes(include=['float64']).columns):
            sc = StandardScaler()
            self.transformations = sc.fit(dataset[dataset.select_dtypes(include=['float64']).columns])
            dataset[dataset.select_dtypes(include=['float64']).columns] = self.transformations.transform(dataset[dataset.select_dtypes(include=['float64']).columns])
        return dataset

    def scaling_test(self,test):
        for column in list(test.select_dtypes(include=['float64']).columns):
            test[test.select_dtypes(include=['float64']).columns]= self.transformations.transform(test[test.select_dtypes(include=['float64']).columns])
        return test
    
    def cleaningNA(self, dataset):
        #Checking missing values:
        self.listOfImportantNan = []
        nanTable = dataset.isna().sum()
        for i, row in enumerate(nanTable):
            index = nanTable.index[i]
            if row > 0:
                table = sm.stats.Table.from_data(dataset[[index,'Churn']])
                rslt = table.test_nominal_association()
                if rslt.pvalue < 0.0025:
                    self.listOfImportantNan.append(index)
                    dataset[index + '_isNan'] = dataset[index].isna()
                if dataset[index].dtypes == 'float64' or dataset[index].dtypes == 'int64':
                    dataset[index] = dataset[index].fillna(dataset[index].median())
                else:
                    dataset[index] = dataset[index].fillna(0)
        return dataset
    
    def cleaningNATest(self, test):
        #Checking missing values:
        for column in test.columns:
            if column in self.listOfImportantNan:
                test[column + '_isNan'] = test[column].isna()
            if test[column].dtypes == 'float64' or test[column].dtypes == 'int64':
                test[column] = test[column].fillna(test[column].median())
            else:
                test[column] = test[column].fillna(0)   
            
        return test    
    
    def visualizingHistAndBar(self, datset):
        #Barcharts
        try:
            for i, col in enumerate(datset.columns):
                try:
                    print("Column",col )
                    plt.figure(i)
                    sns.barplot(x='Churn', y = col, data = datset)
                except Exception as e:
                    print(e)

            #Histograms
            fig = plt.figure(figsize = (25,25))
            ax = fig.gca()
            datset.hist(ax = ax)
        except Exception as e:
            print("Exception visualizingHistAndBar", e);
            
    
    def binning(self, dataset):
        ## Combining too long categorized variables
        for column in list(dataset.select_dtypes(include=['object']).columns):
            if len(dataset[column].value_counts()) > 40:
                firstObesrvations = dataset[column].value_counts()[0:40]
                dataset[column] = dataset[column].apply(lambda x: x if x in firstObesrvations else 'Other')
            dataset[column] = pd.factorize(dataset[column])[0]
        return dataset
    
    def selectingFeatures(self, X,Y):
        clf = ExtraTreesClassifier(n_estimators=56)
        clf = clf.fit(X, Y)
        clf.feature_importances_ 
        model = SelectFromModel(clf, prefit=True)
        X = model.transform(X) 
        return X   
    
    def cleaningOutliers(self, dataset):
        # Deleting outliers for categorical variables
        for column in list(dataset.select_dtypes(include=['float64']).columns):
            threshold=5
            mean_1 = np.mean(dataset[column])
            std_1 =np.std(dataset[column])
            dataset = dataset[dataset[column].apply(lambda x : np.abs((x - mean_1)/std_1)) < threshold]
            
        return dataset
    
    def gettingDecimals(self, row, column):
        if math.isnan(row[column]) != True :
            if (round(float(row[column]))) != float(row[column]):
                return True
            else:
                return False

    def deletingObjectColumns(self, dataset):
        # Deleting outliers for categorical variables
        for column in list(dataset.select_dtypes(include=['object']).columns):
            dataset = dataset.drop([column], axis = 1)
        return dataset
    
    def factorizingCategorical(self, dataset):
        for column in list(dataset.select_dtypes(include=['object']).columns):
            dataset[column] = dataset[column].factorize()[0]
        return dataset
    
    def getLastDigit(self, row):
        return row.name % 2
    
    
    def build_classiffier(self):
        classifier = Sequential()
        classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 62))
        classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 62))
        classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
        classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        return classifier
    
