import numpy as np
from sklearn import cross_validation
from sklearn.svm import SVR
from sklearn import grid_search
from sklearn.externals import joblib
from sklearn.linearmodel import LogisticRegression
from sklearn.feature_selection import VarianceThreshold

# Load data from numpy array 
data_trans = np.load('C:\\Users\\Jacqueline\\Documents\\Dropbox\\SFworksample\\data\\clean_data.npy')
data_labels = np.load('C:\\Users\\Jacqueline\\Documents\\Dropbox\\SFworksample\\data\\clean_labels.npy')

# Split training and test set
data_train, data_test, label_train, label_test = \
cross_validation.train_test_split(data_trans, data_labels, random_state=6)

# Error function
def rmse(pred_labels, true_labels):
    return np.sqrt(((pred_labels - true_labels) ** 2).mean())

selector = VarianceThreshold()
selector.fit(data_train)
selector.variances_

# LET'S MODEL SOME MORE
svr = SVR()
svr.fit(data_train, label_train)
rmse(svr.predict(data_train), label_train)
rmse(svr.predict(data_test), label_test)

# ALLLL THE MODELS
lr = LogisticRegression()
ls.fit(data_train, label_train)
rmse(lr.predict(data_train), label_train)
rmse(svr.predict(data_test), label_test)