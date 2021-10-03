#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 09:29:45 2021

@author: kevinli
"""
#Required Packages
import numpy as np #The Numpy numerical computing library
import pandas as pd #The Pandas data science library
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#Read file 
data=pd.read_csv("NorthwesternCOVID.csv")

#Dropping NA rows 
data.dropna(subset = ['Total Tests'], inplace=True)

#Converting to date time form & renaming the column
data['Week Start']= pd.to_datetime(data['Week Start'], format='%Y-%m-%d')
data.rename(columns={'Week Start': 'Date'}, inplace=True)
data['Date'] = pd.to_datetime(data['Date']).dt.date

#Correlation Plotting
corr = data.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            cmap='coolwarm')
plt.show()

#Line plot for Positivity Rate, Faculty Cases, Student Cases
ax = plt.gca()
formatter = mdates.DateFormatter("%Y-%m-%d")
ax.xaxis.set_major_formatter(formatter)

plt.plot(data["Date"], data["Positivity Rate"])
plt.plot(data["Date"], data["Faculty Cases"])
plt.plot(data["Date"], data["Student Cases"])

#Properly Labelling the Graph


#Preparing for Machine Learning
bins=[]
for i in range (0,6):
    bins.append((np.percentile(data["Positivity Rate"], i*20)))

group_names=[1.0, 2.0, 3.0,4.0,5.0]

data['Positivity Rate'] = pd.cut(data['Positivity Rate'], bins=bins, labels=group_names)

data.dropna(axis = 0, inplace=True)


#Random Forest Classifiers
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


def random_forest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                              random_state = 200)
    clf = RandomForestClassifier(n_estimators=200).fit(X_train, y_train)
    print('Accuracy of Random Forest classifier on training set: {:.2f}'
          .format(clf.score(X_train, y_train)))
    print('Accuracy of Random Forest classifier on test set: {:.2f}'
          .format(clf.score(X_test, y_test)))
    
target = data['Positivity Rate']
target = pd.get_dummies(target)
# Removing the columns Player Name, Wins, and Winner from the dataframe
ml_df = data.copy()
ml_df.drop(['Date'], axis=1, inplace=True)
print(random_forest(ml_df, target))



