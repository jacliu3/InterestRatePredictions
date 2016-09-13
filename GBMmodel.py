import numpy as np
from sklearn import cross_validation
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import grid_search
from sklearn.externals import joblib

# Load data from numpy array 
data_trans = np.load('C:\\Users\\Jacqueline\\Documents\\Dropbox\\SFworksample\\data\\clean_data.npy')
data_labels = np.load('C:\\Users\\Jacqueline\\Documents\\Dropbox\\SFworksample\\data\\clean_labels.npy')

# Split training and test set
data_train, data_test, label_train, label_test = \
cross_validation.train_test_split(data_trans, data_labels, random_state=3)
label_train = np.array(label_train)
label_test = np.array(label_test)

# Error function
def rmse(pred_labels, true_labels):
    return np.sqrt(((pred_labels - true_labels) ** 2).mean())
    
# LET'S BEGIN MODELING (finally)
# Creating a gradient boosted regressor
gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                max_depth=3, loss='ls')
gbr.fit(data_train, label_train)

# Feature selection based on importance (assigned by model)
# (tried kernel PCA but ran out of memory)
# (tried feature elimination but takes too long)
use_feature = gbr.feature_importances_ > 0

trim_train = data_train[:, use_feature]
trim_test = data_test[:, use_feature]

# Tuning the hyperparameters
param = {'max_features':['sqrt', 'log2'], 'subsample':[0.3, 0.5, 0.75, 0.8, 0.9]}
clf = grid_search.GridSearchCV(gbr, param)
clf.fit(trim_train, label_train)
max_features = clf.best_params_['max_features']
subsample = clf.best_params_['subsample']

# Yes, I should've done grid search on all parameters at once
# but I don't think my laptop could've handled it :/
# Maybe it's time to use those free AWS credits
gbr = GradientBoostingRegressor(n_estimators=100, loss='ls',
                                max_features=max_features, subsample=subsample)
param = {'learning_rate':[0.5, 0.25, 0.1, 0.01, 0.001], 'max_depth':[5, 7, 10, 15]}
clf = grid_search.GridSearchCV(gbr, param)
clf.fit(trim_train, label_train)
learning_rate = clf.best_params_['learning_rate']
max_depth = clf.best_params_['max_depth']

# Evaluation of best model
gbr = clf.best_estimator_
pred = gbr.predict(trim_test)
rmse = rmse(pred, label_test)

# Save model and other important info
joblib.dump(gbr, 'C:\\Users\\Jacqueline\\Documents\\Dropbox\\SFworksample\\models\\gbr.pkl')
np.save('C:\\Users\\Jacqueline\\Documents\\Dropbox\\SFworksample\\models\\gbr_features.npy', use_feature)
