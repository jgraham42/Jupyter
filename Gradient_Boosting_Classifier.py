# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 15:15:03 2019

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

Y = data['target2']

X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.3,
                                                    random_state=1)

names = X_train.columns


gbc = GradientBoostingClassifier(learning_rate = 0.045,
                                 n_estimators = 3000,
                                 min_samples_leaf = 10,
                                 max_depth = 3,)
#gbc = LogisticRegression(penalty='l2')

gbc.fit(X_train,Y_train)
#gbc.fit(X_train, Y_train)

fpr, tpr, thresholds = roc_curve(Y_test, gbc.predict(X_test))
print("Accuracy on training set: {:.3f}".format(gbc.score(X_train, Y_train)))
print("Accuracy on test set: {:.3f}".format(gbc.score(X_test, Y_test)))
#print("GBC Feature importances:\n{}".format(gbc.feature_importances_))
print("AUC: {:.3f}".format(auc(fpr,tpr)))
print("GBC Matthews Correlation: {:.3f}".format(matthews_corrcoef(Y_test,gbc.predict(X_test))))
print(classification_report(Y_test,gbc.predict(X_test)))
print('Confusion Matrix:\n TN     FP \n FN     TP \n',confusion_matrix(Y_test,gbc.predict(X_test)))
#%%
f_imp = pd.DataFrame(np.vstack(gbc.feature_importances_).T, columns=names).T.sort_values(by=0, ascending=False)
f_imp.head(3)

top_n = f_imp.reset_index()['index'].values[0]
choose_feature = 'ComparableCredit'

a = X_train.columns.get_loc(choose_feature)
features = [a]
feat_num = a

#fig, axs = plot_partial_dependence(gbc, X_train, features,
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
#
#plot_feature_importances(gbc)
#
#plt.show()

pd.DataFrame(np.vstack(gbc.feature_importances_).T, columns=names).T.sort_values(by=0).plot(kind='barh')

pdep = partial_dependence(gbc, features, X = X_train, grid_resolution=100)

columns_new = ['Risk Probabilities',names[feat_num]]
fico = pd.DataFrame(np.vstack(pdep).T, columns = columns_new)

fico = fico[fico[names[feat_num]]>600]

fico.plot(names[feat_num],'Risk Probabilities')

# ROC Curve Plot
logit_roc_auc = roc_auc_score(Y_test, gbc.predict(X_test))
fpr, tpr, thresholds = roc_curve(Y_test, gbc.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Gradient Boosting Classifier (area = %0.2f)' % logit_roc_auc)
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
#len(X_train[(X_train['truck_age_at_orig']>5) & (X_train['truck_age_at_orig']<20)])
##%%
#list(X_train)
##%%
#norm = Normalizer()
#data = pd.DataFrame(norm.fit_transform(data),columns=data.columns)
##%%
#pd.DataFrame(np.vstack(pdep).T, columns = ['value',names[feat_num]])
sorted(X)