# -*- coding: utf-8 -*-
"""
Justin Clark
CSYS 300
Final Project
popularityPrediction.py

Use different ML methods to predict song popularity 
Outline:
    
"""

### 1. Imports ###
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from sklearn import datasets
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

from collections import Counter
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
from scipy.stats import randint as sp_randint

from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor 
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn import tree
from mlens.visualization import corrmat

from sklearn.neural_network import MLPRegressor

#data = pd.read_csv('Project/rap_1993-2019.csv')
data = pd.read_csv('rap_1993_2020.csv')
data = data.rename(columns = {'Prop Lines Neg': 'prop_neg',
                              'Prop Lines Neu': 'prop_neut',
                              'Prop Lines Pos': 'prop_pos',
                              'Avg Sentiment': 'avg_sent',
                              'Year': 'year',
                              'Word Count': 'word_count',
                              'Prop Unique Words': 'p_unique'})
data = data[data['popularity'] != 0]

data['class'] = data['popularity']>50
data['class'] = data['class'].astype(int)
Counter(data['class'].tolist())


data.describe().T.iloc[0:14,0:3]

rows = data.shape[0]
cols = data.shape[1]
target_index = data.columns.get_loc("popularity")
X = data.iloc[:,target_index + 1:cols-1]
feature_names = X.columns
Y = data.iloc[:,-1]
X = np.matrix(X)
Y = np.array(Y).T

# Distribution of Target Values
avg_pop = np.mean(Y)
std = np.std(Y)
plt.hist(Y,bins = 50)
plt.text(20,31,"Mean: {:.2f} Std: {:.2f}".format(avg_pop,std),fontsize = 14)
plt.grid(axis = 'y',alpha = 0.75)
plt.xlabel("Target Value: Song Popularity Score",fontsize = 18)
plt.ylabel("Frequency",fontsize = 18)
plt.title("Distribution of Target Values: Song Popularity Scores",fontsize = 18)
plt.show()


#X = preprocessing.standardize(X)
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.2,random_state = 1)
#X_train,X_valid,y_train,y_valid = train_test_split(X_train,y_train,test_size = 0.2)

sc  = StandardScaler()
X_standardized = sc.fit_transform(X)
X_train_standardized= sc.fit_transform(X_train)
X_test_standardized = sc.fit_transform(X_test)

C_list = [10,1,.1,.001]

for reg_penalty in C_list:
    clf = LogisticRegression(penalty = 'l1',C=reg_penalty,solver = 'liblinear')
    clf.fit(X_train_standardized,y_train)
    feature_importance = clf.coef_[0]
    y_pred = clf.predict(X_test_standardized)
    confusion_matric = metrics.confusion_matrix(y_test,y_pred)
    f1_score = metrics.f1_score(y_test,y_pred)
    print("Regularization Pentality: {}".format(reg_penalty))
    print("Feature Coefficients: {}".format(clf.coef_))
    print("Training Accuracy: {}".format(clf.score(X_train_standardized,y_train)))
    print("Testing Accuracy: {}".format(clf.score(X_test_standardized,y_test)))
    print("F1 Score: {}".format(f1_score))
    for i,v in enumerate(feature_importance):
        print("Feature: {} Importancce: {}".format(feature_names[i],v))
    print(confusion_matrix)
    print(metrics.classification_report(y_test,y_pred))
    print(metrics.precision_recall_fscore_support(y_test, y_pred, average='macro'))
    AUC = metrics.roc_auc_score(y_test,y_pred)
    print("AUC: {}".format(AUC))

    print("-"*100)

#######################################
 #SVM
#######################################
model = SVC()
model.fit(X_train_standardized,y_train)
y_pred = model.predict(X_test_standardized)
print(metrics.classification_report(y_test,y_pred))


param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              #'kernel': ['rbf']}
              'kernel': ['rbf']} 
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3) 
grid.fit(X_train, y_train)
print(grid.best_params_) 
print(grid.best_estimator_) 
grid_predictions = grid.predict(X_test)
print(metrics.classification_report(y_test, grid_predictions))
print(metrics.precision_recall_fscore_support(y_test, grid_predictions, average='macro'))
AUC = metrics.roc_auc_score(y_test,grid_predictions)
print("AUC: {}".format(AUC))

 


######################################
# Decision Tree / Random Forest
######################################

print("-"*100)

#### Tree based feature selection
forest = ExtraTreesClassifier(n_estimators = 250)
forest.fit(X, Y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)

# Plot the feature importances of the forest
plt.figure()
plt.title("Extra Classifers: Feature Importances")
plt.barh(range(X.shape[1]), importances[indices],
       color="grey",edgecolor = 'black', xerr=std[indices],ecolor = 'black', align="center")
# If you want to define your own labels,
# change indices to a list of labels on the following line.
plt.yticks(range(X.shape[1]), feature_names[indices])
plt.ylim([-1, X.shape[1]])
plt.tight_layout()
plt.savefig(os.getcwd() + '/Plots/feature_importance_tree.png',dpi = 900)
plt.show()
# display the relative importance of each attribute
print(forest.feature_importances_)
model = SelectFromModel(forest,prefit = True)
X_feature_selection = model.transform(X)
print(X_feature_selection.shape)
print("-"*100)

Xf_train, Xf_test, yf_train, yf_test = train_test_split(X_feature_selection,Y,test_size = 0.2,random_state = 1)

#Single Decision Tree: No Feature Selection
clf = DecisionTreeClassifier()
clf.fit(X_train_standardized,y_train)
y_pred = clf.predict(X_test_standardized)
print("Single Decision tree")
print(metrics.classification_report(y_test,y_pred))
f1_score = metrics.f1_score(y_test,y_pred)
print(metrics.precision_recall_fscore_support(y_test, y_pred, average='macro'))
print("F1 Score: {}".format(f1_score))
AUC = metrics.roc_auc_score(y_test,y_pred)
print("AUC: {}".format(AUC))

print("-"*100)


# Single Decision Tree: Feature Selection
clf = DecisionTreeClassifier()
clf.fit(Xf_train,yf_train)
y_pred = clf.predict(Xf_test)
print("Single Decision tree:Feature Selection")
print(metrics.classification_report(yf_test,y_pred))
f1_score = metrics.f1_score(y_test,y_pred)
print("F1 Score: {}".format(f1_score))
print(metrics.precision_recall_fscore_support(y_test, y_pred, average='macro'))
AUC = metrics.roc_auc_score(y_test,y_pred)
print("AUC: {}".format(AUC))
print("-"*100)


# Random Forest: No Feature Selection
num_trees = 1000
clf = RandomForestClassifier(n_estimators = num_trees,bootstrap = True,max_features = 'sqrt')
clf.fit(X_train_standardized,y_train)
y_pred = clf.predict(X_test_standardized)
print("Random Forest")
print(metrics.classification_report(y_test,y_pred))
f1_score = metrics.f1_score(y_test,y_pred)
print("F1 Score: {}".format(f1_score))
print(metrics.precision_recall_fscore_support(y_test, y_pred, average='macro'))
AUC = metrics.roc_auc_score(y_test,y_pred)
print("AUC: {}".format(AUC))
print("-"*100)


# Random Forest: Feature Selection
print("Random Forest:Feature Selection")
clf = RandomForestClassifier(n_estimators = num_trees,bootstrap = True,max_features = 'sqrt')
clf.fit(Xf_train,yf_train)
y_pred = clf.predict(Xf_test)
print(metrics.classification_report(yf_test,y_pred))
f1_score = metrics.f1_score(y_test,y_pred)
print("F1 Score: {}".format(f1_score))
print(metrics.precision_recall_fscore_support(y_test, y_pred, average='macro'))
AUC = metrics.roc_auc_score(y_test,y_pred)
print("AUC: {}".format(AUC))
print("-"*100)



