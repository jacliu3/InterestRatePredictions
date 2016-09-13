import numpy as np
from sklearn import cross_validation
from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression

# Load data from numpy array 
data_trans = np.load('C:\\Users\\Jacqueline\\Documents\\Dropbox\\SFworksample\\data\\clean_data.npy')
data_labels = np.load('C:\\Users\\Jacqueline\\Documents\\Dropbox\\SFworksample\\data\\clean_labels.npy')

# Split training and test set
data_train, data_test, label_train, label_test = \
cross_validation.train_test_split(data_trans, data_labels, random_state=6)

# Error function
def rmse(pred_labels, true_labels):
    return np.sqrt(((pred_labels - true_labels) ** 2).mean())

# ALLLL THE MODELS
# JK. Saddest thing of my life
# After all that time spent with the SVRs, KernelRidge, etc...
lr = LinearRegression()
lr.fit(data_train, label_train)
best = rmse(lr.predict(data_test), label_test)

# Save
joblib.dump(lr, 'C:\\Users\\Jacqueline\\Documents\\Dropbox\\SFworksample\\models\\lr.pkl')

