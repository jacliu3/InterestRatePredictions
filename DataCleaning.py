import csv
import re    
from calendar import month_abbr
import numpy as np
from sklearn import feature_extraction, preprocessing, cross_validation 

# PREPROCESSING AND CLEANING TRAINING DATA
data_file = open(
    'C:\\Users\\Jacqueline\\Documents\\Dropbox\\SFworksample\\Data for Cleaning & Modeling.csv')
data_dict = csv.DictReader(data_file)
data_list = []
data_labels = []

months = list(month_abbr)
grade_levels = {'A':0, 'B':5, 'C':10, 'D':15, 'E':20, 'F':25, 'G':30}

# Cleaning each sample
# (removing values, casting to integers, etc...)
for row in data_dict:
    # Find interest rate as a float
    # If can't, discard this sample
    try:
        label = float(re.search('[0-9.]+',row['X1']).group(0))
    except AttributeError:
        continue
    
    row.pop("X1", None)           #interest rate shouldn't be a feature
    row.pop("X2", None)           #loan id is not informative
    row.pop("X3", None)           #hmmmm don't want to rely on this
    
    requested = row.pop("X4")
    if requested:
        requested = re.search('[0-9,]+', requested).group(0)
        requested = int(requested.replace(',', ''))
    row['requested'] = requested
    
    funded = row.pop("X5")
    if funded:
        funded = re.search('[0-9,]+', funded).group(0)
        funded = int(funded.replace(',', ''))
    row['funded'] = funded
    
    investor = row.pop("X6")
    if investor:
        investor = re.search('[0-9,]+', investor).group(0)
        investor = int(investor.replace(',', ''))
    row['investor'] = investor
    
    row['num_payments'] = row.pop("X7")
    
    # Convert grade/subgrade into a continuous variable
    # which I hope makes sense, given it's a measure of risk
    # but then again I don't study business/finance :/
    # Actuary would probably kill me
    grade = row.pop("X8")
    subgrade = row.pop("X9")
    if subgrade:
        grade = re.search('[A-z]', subgrade).group(0)
        grade = grade_levels[grade] + int(re.search('[0-9]', subgrade).group(0))
    elif grade:
        grade = grade_levels[grade] + 2.5
    row['grade'] = grade
    
    row['job'] = 1 if row.pop("X10", None) else 0
    
    years_employed = row.pop("X11", None)
    if years_employed:
        if re.search('<', years_employed):
            years_employed = 0
        elif re.search('n/a', years_employed):
            years_employed = -1
        else:
            try:
                years_employed = re.search('[0-9]', years_employed).group(0)
                years_employed = int(years_employed.replace(',', ''))
            except AttributeError:
                years_employed = None
    row['years_employed'] = years_employed
    
    row['home'] = row.pop("X12")
    
    income = row.pop("X13")       #only cast if not None
    row['income'] = float(income) if income else None 
    
    row['income_check'] = row.pop("X14")

    row.pop("X15")        #are interest rates supposed to be seasonal?...
    row.pop("X16")        #too much text -- can do analytics on this later
                          #(bag of words, term frequency inverse document frequecy)
                          
    row['category'] = row.pop("X17")
    
    row.pop("X18", None)  #loan title 
    row.pop("X19", None)  #zip code -- categorical but way too many to binarize
    
    row['state'] = row.pop("X20")
    
    debt_ratio = row.pop("X21")       #only cast if not None
    row['debt_ratio'] = float(debt_ratio) if debt_ratio else None
    
    past_dues = row.pop("X22")        #only cast if not None
    row['past_dues'] = int(past_dues) if past_dues else None
    
    credit_hist = row.pop("X23")
    if credit_hist:
        year = int(re.search('[0-9]{2}', credit_hist).group(0))
        year = (year + 100) * 100 if year < 17 else year * 100
        month = months.index(re.search('[A-z]{3}', credit_hist).group(0))
        credit_hist = year + month
    row['credit_hist'] = credit_hist
    
    inquiries = row.pop("X24")         #only cast if not None
    row['inquiries'] = int(inquiries) if inquiries else None

    delinquency = row.pop("X25")       #only cast if not None
    row['delinquency'] = int(delinquency) if delinquency else None
    
    last_record = row.pop("X26")       #only cast if not None
    row['last_record'] = int(last_record) if last_record else None
    
    lines = row.pop("X27")             #only cast if not None
    row['lines'] = int(lines) if lines else None
    
    records = row.pop("X28")           #only cast if not None
    row['records'] = int(records) if records else None
    
    credit_bal = row.pop("X29")        #only cast if not None
    row['credit_bal'] = int(credit_bal) if credit_bal else None
    
    credit_use = row.pop("X30")
    if credit_use:
        try:
            credit_use = float(re.search('[0-9]', credit_use).group(0))
        except AttributeError:
            credit_use = None
    else:
        credit_use = None
    row['credit_use'] = credit_use
    
    credit_lines = row.pop("X31")      #only cast if not None
    row['credit_lines'] = int(credit_lines) if credit_lines else None
    
    row['status'] = row.pop("X32")
    
    # Don't add entries with over 10 missing values 
    if sum(val == None or val == '' for val in list(row.viewvalues())) < 10:
        row.update((k, None) for k in row if row[k] == '')
        data_list.append(row)
        data_labels.append(label)
    
v = feature_extraction.DictVectorizer(sparse=False)
data_samples = v.fit_transform(data_list)
data_labels = np.array(data_labels)

# Replace 'nan' values with 1 for categorical variables,
# specifically for the columns that represent missing values
features = v.get_feature_names()
i = features.index("home")       #home ownership 
data_samples[np.isnan(data_samples[:,i]), i] = 1 
i = features.index("income")     #income verification
data_samples[np.isnan(data_samples[:,i]), i] = 1

# The columns 'delinquency' and 'last_record' have over
# 50% missing values, so rather than trying to impute 
# those (risky business) we'll binarize the column. 
i = features.index("delinquency")    #months since last delinquency
data_samples[:, i] = np.isnan(data_samples[:, i]).astype(int)
i = features.index("last_record")    #months since last public record
data_samples[:, i] = np.isnan(data_samples[:, i]).astype(int)

# For the remaining missing values, we'll impute them 
# with the median of the column (means make me nervous).
# Tbh it'd be interesting to use kNN instead, but sadly enough
# the package 'fancyimpute' won't install properly, plus
# that computation might push my dying laptop over the edge.
imp = preprocessing.Imputer(axis=0, strategy='median', missing_values="NaN")
data_trans = imp.fit_transform(data_samples)

# APPLYING SAME TRANSFORMATIONS ON HOLDOUT DATA
data_file = open(
    'C:\\Users\\Jacqueline\\Documents\\Dropbox\\SFworksample\\Holdout for Testing.csv')
data_dict = csv.DictReader(data_file)
data_list = []
data_labels = []

months = list(month_abbr)
grade_levels = {'A':0, 'B':5, 'C':10, 'D':15, 'E':20, 'F':25, 'G':30}

# Cleaning each sample
# (removing values, casting to integers, etc...)
for row in data_dict:
    # Find interest rate as a float
    # If can't, discard this sample
    try:
        label = float(re.search('[0-9.]+',row['X1']).group(0))
    except AttributeError:
        continue
    
    row.pop("X1", None)           #interest rate shouldn't be a feature
    row.pop("X2", None)           #loan id is not informative
    row.pop("X3", None)           #hmmmm don't want to rely on this
    
    requested = row.pop("X4")
    if requested:
        requested = re.search('[0-9,]+', requested).group(0)
        requested = int(requested.replace(',', ''))
    row['requested'] = requested
    
    funded = row.pop("X5")
    if funded:
        funded = re.search('[0-9,]+', funded).group(0)
        funded = int(funded.replace(',', ''))
    row['funded'] = funded
    
    investor = row.pop("X6")
    if investor:
        investor = re.search('[0-9,]+', investor).group(0)
        investor = int(investor.replace(',', ''))
    row['investor'] = investor
    
    row['num_payments'] = row.pop("X7")
    
    # Convert grade/subgrade into a continuous variable
    # which I hope makes sense, given it's a measure of risk
    # but then again I don't study business/finance :/
    # Actuary would probably kill me
    grade = row.pop("X8")
    subgrade = row.pop("X9")
    if subgrade:
        grade = re.search('[A-z]', subgrade).group(0)
        grade = grade_levels[grade] + int(re.search('[0-9]', subgrade).group(0))
    elif grade:
        grade = grade_levels[grade] + 2.5
    row['grade'] = grade
    
    row['job'] = 1 if row.pop("X10", None) else 0
    
    years_employed = row.pop("X11", None)
    if years_employed:
        if re.search('<', years_employed):
            years_employed = 0
        elif re.search('n/a', years_employed):
            years_employed = -1
        else:
            try:
                years_employed = re.search('[0-9]', years_employed).group(0)
                years_employed = int(years_employed.replace(',', ''))
            except AttributeError:
                years_employed = None
    row['years_employed'] = years_employed
    
    row['home'] = row.pop("X12")
    
    income = row.pop("X13")       #only cast if not None
    row['income'] = float(income) if income else None 
    
    row['income_check'] = row.pop("X14")

    row.pop("X15")        #are interest rates supposed to be seasonal?...
    row.pop("X16")        #too much text -- can do analytics on this later
                          #(bag of words, term frequency inverse document frequecy)
                          
    row['category'] = row.pop("X17")
    
    row.pop("X18", None)  #loan title 
    row.pop("X19", None)  #zip code -- categorical but way too many to binarize
    
    row['state'] = row.pop("X20")
    
    debt_ratio = row.pop("X21")       #only cast if not None
    row['debt_ratio'] = float(debt_ratio) if debt_ratio else None
    
    past_dues = row.pop("X22")        #only cast if not None
    row['past_dues'] = int(past_dues) if past_dues else None
    
    credit_hist = row.pop("X23")
    if credit_hist:
        year = int(re.search('[0-9]{2}', credit_hist).group(0))
        year = (year + 100) * 100 if year < 17 else year * 100
        month = months.index(re.search('[A-z]{3}', credit_hist).group(0))
        credit_hist = year + month
    row['credit_hist'] = credit_hist
    
    inquiries = row.pop("X24")         #only cast if not None
    row['inquiries'] = int(inquiries) if inquiries else None

    delinquency = row.pop("X25")       #only cast if not None
    row['delinquency'] = int(delinquency) if delinquency else None
    
    last_record = row.pop("X26")       #only cast if not None
    row['last_record'] = int(last_record) if last_record else None
    
    lines = row.pop("X27")             #only cast if not None
    row['lines'] = int(lines) if lines else None
    
    records = row.pop("X28")           #only cast if not None
    row['records'] = int(records) if records else None
    
    credit_bal = row.pop("X29")        #only cast if not None
    row['credit_bal'] = int(credit_bal) if credit_bal else None
    
    credit_use = row.pop("X30")
    if credit_use:
        try:
            credit_use = float(re.search('[0-9]', credit_use).group(0))
        except AttributeError:
            credit_use = None
    else:
        credit_use = None
    row['credit_use'] = credit_use
    
    credit_lines = row.pop("X31")      #only cast if not None
    row['credit_lines'] = int(credit_lines) if credit_lines else None
    
    row['status'] = row.pop("X32")
    
    # Don't add entries with over 10 missing values 
    if sum(val == None or val == '' for val in list(row.viewvalues())) < 10:
        row.update((k, None) for k in row if row[k] == '')
        data_list.append(row)
        data_labels.append(label)

np.save('C:\\Users\\Jacqueline\\Documents\\Dropbox\\SFworksample\\clean_data', data_trans)


# DIMENSIONALITY REDUCTION


# CREATING THE MODELS
# Split train and test set
data_train, data_test, label_train, label_test = \
cross_validation.train_test_split(data_trans, data_labels, random_state=3)

# Error function
def rmse(pred_labels, true_labels):
    np.sqrt(((pred_labels - true_labels) ** 2).mean())

# Let's make a GBM
from sklearn.ensemble import GradientBoostingRegressor
gbm = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3,
                                random_state=33, loss='ls').fit(data_train, label_train)
                                
# Probably ought to tune the hyperparameters
from sklearn import grid_search
param = {'learning_rate':[0.5, 0.25, 0.1, 0.01, 0.001], 'max_depth':[1, 2, 3, 5, 10]}
gbm = GradientBoostingRegressor()
clf = grid_search.GridSearchCV(gbm, param)
clf.fit(data_train, label_train)

>>> from sklearn import svm, grid_search, datasets
>>> iris = datasets.load_iris()
>>> parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
>>> svr = svm.SVC()
>>> clf = grid_search.GridSearchCV(svr, parameters)
>>> clf.fit(iris.data, iris.target)
...                             
GridSearchCV(cv=None, error_score=...,
       estimator=SVC(C=1.0, cache_size=..., class_weight=..., coef0=...,
                     decision_function_shape=None, degree=..., gamma=...,
                     kernel='rbf', max_iter=-1, probability=False,
                     random_state=None, shrinking=True, tol=...,
                     verbose=False),
       fit_params={}, iid=..., n_jobs=1,
       param_grid=..., pre_dispatch=..., refit=...,
       scoring=..., verbose=...)


    