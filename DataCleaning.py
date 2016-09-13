import csv
import re    
from calendar import month_abbr
import numpy as np
from sklearn import feature_extraction, preprocessing, cross_validation 

# Useful mappings
months = list(month_abbr)
grade_levels = {'A':0, 'B':5, 'C':10, 'D':15, 'E':20, 'F':25, 'G':30}

# PREPROCESSING AND CLEANING DATA
# (both training and holdout)

data_file = open(
    'C:\\Users\\Jacqueline\\Documents\\Dropbox\\SFworksample\\data\\raw_data.csv')
data_dict = csv.DictReader(data_file)

hold_file = open(
    'C:\\Users\\Jacqueline\\Documents\\Dropbox\\SFworksample\\data\\raw_holdout.csv')
hold_dict = csv.DictReader(hold_file)

def read_dict(data_dict, train=True):
    data_list = []
    data_labels = []
    
    # Cleaning each sample
    # (removing values, casting to integers, etc...)
    for row in data_dict:
        
        # Find interest rate as a float
        # If can't, discard this sample
        try:
            label = float(re.search('[0-9.]+',row['X1']).group(0))
        except AttributeError:
            # If there's no usable interest rate, can't train on this sample
            if train:
                continue
            else:
                label = None
        
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
            year = int(re.search('[0-9]+', credit_hist).group(0))
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
        
        # Don't add entries with over 10 missing values for the training set
        if not train or sum(val == None or val == '' for val in list(row.viewvalues())) < 10:
            row.update((k, None) for k in row if row[k] == '')
            data_list.append(row)
            data_labels.append(label)
            
    return data_list, data_labels
    
data_list, data_labels = read_dict(data_dict)
data_labels = np.array(data_labels)
hold_list, hold_labels = read_dict(hold_dict, train=False)

# Vectorizing categorical variables
# Shaping data as numpy array
v = feature_extraction.DictVectorizer(sparse=False)
data_samples = v.fit_transform(data_list)
hold_samples = v.transform(hold_list)

def clean_samples(data_samples):
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
    
    return data_samples

data_samples = clean_samples(data_samples)
hold_samples = clean_samples(hold_samples)

# For the remaining missing values, we'll impute them 
# with the median of the column (means make me nervous).
# Tbh it'd be interesting to use kNN instead, but sadly enough
# the package 'fancyimpute' won't install properly, plus
# that computation might push my dying laptop over the edge.
imp = preprocessing.Imputer(axis=0, strategy='median', missing_values="NaN")
data_trans = imp.fit_transform(data_samples)
hold_trans = imp.transform(hold_samples)

# Save the numpy arrays of data
np.save('C:\\Users\\Jacqueline\\Documents\\Dropbox\\SFworksample\\clean_data', data_trans)
np.save('C:\\Users\\Jacqueline\\Documents\\Dropbox\\SFworksample\\clean_holdout', hold_trans)
