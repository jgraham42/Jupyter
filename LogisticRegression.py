# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 09:13:56 2019

@author: jgraham
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble.partial_dependence import partial_dependence
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer, MaxAbsScaler, MinMaxScaler
from sklearn.preprocessing import Imputer
from sklearn.metrics import classification_report, auc, confusion_matrix, roc_curve
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

data = pd.read_csv('c:/users/jgraham/desktop/data robot model sets/dataset10.csv')
data = data.drop(['SmeScore'], axis=1)

#### Imputation ####

# Impute with zero
data = data.fillna(0)

# Drop all rows with missing values
#data = data.dropna()

# Impute with mean, median, or most frequent value (aka. mode) 
imp = Imputer(strategy = 'most_frequent', verbose = 1)
#data = pd.DataFrame(imp.fit_transform(data),columns=data.columns)


X = data.drop(['target',
               'target2',
               'late_days',
               'totalLoss',
               'Unnamed: 0',
#               'term',
               'origAmt',
               'DrivingExperience',
#               'LTV',
#               'LatePaymentCount',
#               'PaynetMasterscore',
#               'RealEstatePayment',
#               'ResidentStability',
#               'RevolvingBalance',
#               'TradeLines',
#               'YearsInBureau',
#               'YearsInBusiness',
#               'new',
#               'used',
#               'truck',
#               'trailer',
#               'truck_age_at_orig',
#               'orig_amt_>150k'
               ], axis=1)

X = sm.add_constant(X)

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

Y = data['target']

X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.3,
                                                    random_state=1)

names = X_train.columns

logit = sm.Logit(Y,X)

result = logit.fit()

print('Summary: \n',result.summary())

print('Confidence Intervals: \n', result.conf_int())

print('Odds Ratios: \n', np.exp(result.params))

print('Coefficients: \n', result.params)

logreg = LogisticRegression(class_weight = 'balanced')
logreg.fit(X_train,Y_train)
y_pred = logreg.predict(X_test)
print('Accuracy:  ',logreg.score(X_test,Y_test))

confusion_matrix = confusion_matrix(Y_test, y_pred)
print('Confusion Matrix: \n',confusion_matrix)

logit_roc_auc = roc_auc_score(Y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(Y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
#%%
fig = plt.figure(figsize=(8,20))
fig = sm.graphics.plot_partregress_grid(result, fig=fig)
