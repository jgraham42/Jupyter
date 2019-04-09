# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 13:26:47 2019

@author: jgraham
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble.partial_dependence import partial_dependence
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer, MaxAbsScaler, MinMaxScaler
from sklearn.preprocessing import Imputer
from sklearn import metrics

data = pd.read_csv('c:/users/jgraham/desktop/data robot model sets/dataset10.csv')
data = data.drop(['SmeScore'], axis=1)

#### Imputation ####

# Impute with zero
#data = data.fillna(0)

# Drop all rows with missing values
data = data.dropna()

# Impute with mean, median, or most frequent value (aka. mode) 
imp = Imputer(strategy = 'most_frequent', verbose = 1)
#data = pd.DataFrame(imp.fit_transform(data),columns=data.columns)


X = data.drop(['target',
               'target2',
               'late_days',
               'totalLoss',
               'Unnamed: 0',
               'origAmt',
               'term'], axis=1)

#### Normalization ####

# Unit Norm
norm = Normalizer()
#X = pd.DataFrame(norm.fit_transform(X),columns=X.columns)

## Maximum absolute value
max_abs = MaxAbsScaler()
#X = pd.DataFrame(max_abs.fit_transform(X),columns=X.columns)
#
## MinMax Scaling
minmax = MinMaxScaler()
#X = pd.DataFrame(minmax.fit_transform(X),columns=X.columns)

Y = data['totalLoss']

X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.2,
                                                    random_state=1)

names = X_train.columns


gbr = GradientBoostingRegressor(n_estimators = 1500)
#logreg = LogisticRegression(penalty='l2')

gbr.fit(X_train,Y_train)
#logreg.fit(X_train, Y_train)


print("Accuracy on training set: {:.3f}".format(gbr.score(X_train, Y_train)))
print("Accuracy on test set: {:.3f}".format(gbr.score(X_test, Y_test)))
#print("Gradient Boosting Regressor Feature importances:\n{}".format(gbr.feature_importances_))

#print("gbr Matthews Correlation: {:.3f}".format(matthews_corrcoef(Y_test,gbr.predict(X_test))))

f_imp = pd.DataFrame(np.vstack(gbr.feature_importances_).T, columns=names).T.sort_values(by=0, ascending=False)
f_imp.head(3)

a = X_train.columns.get_loc(f_imp.reset_index()['index'].values[0])
features = [a]
feat_num = a

#fig, axs = plot_partial_dependence(gbr, X_train, features,
#                                       feature_names=names, grid_resolution=50)
#plt.subplots_adjust(top=0.9)  # tight_layout causes overlap

#def plot_feature_importances(model):
#    plt.figure(figsize=(8,6))
#    n_features = len(names)
#    plt.barh(range(n_features), model.feature_importances_, align='center')
#    plt.yticks(np.arange(n_features), names)
#    plt.xlabel("Feature importance")
#    plt.ylabel("Feature")
#    plt.ylim(-1, n_features)

pd.DataFrame(np.vstack(gbr.feature_importances_).T, columns=names).T.sort_values(by=0).plot(kind='barh')

#plot_feature_importances(gbr)
#
#plt.show()

pdep = partial_dependence(gbr, features, X = X_train, grid_resolution=50)

columns_new = ['Risk Probabilities',names[feat_num]]
fico = pd.DataFrame(np.vstack(pdep).T, columns = columns_new)

#fico = fico[fico[names[feat_num]]>500]

fico.plot(names[feat_num],'Risk Probabilities')
#%%
len(X_train[(X_train['truck_age_at_orig']>5) & (X_train['truck_age_at_orig']<20)])
#%%
list(X_train)
#%%