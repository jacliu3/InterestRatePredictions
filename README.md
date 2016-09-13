# Interest Rate Predictions for Loan Applications
### Sept 13, 2016
### Work sample (yay jobs)

Given ~350k samples of customers applying for loans, predict the interest rate using two different ML algorithms. See Metadata.csv for variable information.

In the data cleaning processing (DataCleaning.py), categorical variables were vectorized while text fields were omitted; given more time, more feature-engineering work could be done with text analytics (bag-of-words, term frequency inverse doc frequency). Features with a majority of missing values were binarized (value or none) while other missing values were imputed with the median of the column (kNN would be an interesting method if I had more computing power). Samples with too many missing features were removed.

I used an ensemble of gradient-boosted regression trees and a simple linear regression; the models are saved in the *models* folder while the trained models were pickled in the *saved_models* folder. 
