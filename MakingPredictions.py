import numpy as np
from sklearn.externals import joblib

# Load model and data
gbr = joblib.load('C:\\Users\\Jacqueline\\Documents\\Dropbox\\SFworksample\\models\\gbr.pkl')
use_feature = np.load('C:\\Users\\Jacqueline\\Documents\\Dropbox\\SFworksample\\models\\gbr_features.npy')
hold_trans = np.load('C:\\Users\\Jacqueline\\Documents\\Dropbox\\SFworksample\\data\\clean_holdout.npy')
hold_trim = hold_trans[:, use_features]

# Make predictions wOoooOOOOoooOOoOOow
gbr_pred = gbr.predict(hold_trim)