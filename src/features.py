import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    A feature engineering class for time series forecasting,

    Params:
    target_col (str): The column name of the target variable.

    Methods:
    fit(X, y=None): Fit the transformer to the data.
    transform(X): Transform the input data. Addiing Lag, Rolling, & Safe Percentage Change features.

    Returns:
    X_new (pd.DataFrame): The transformed feature-engineered data.
    """

    def __init__(self, target_col='T_(degC)'):
          self.target_col = target_col
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_new = X.copy()
        target = X_new[self.target_col]
        
        # ----- Lag Features -----
        for days in [1, 7, 14]:  # [1, 7,14] day lags
            X_new[f'{self.target_col}_lag_{days}d'] = target.shift(days*144)  
            
        # ----- Rolling Features -----
        for days in [1, 7, 14]:
            window = days*144
            X_new[f'{self.target_col}_rollmean_{days}d'] = target.rolling(window, min_periods=1).mean()
            X_new[f'{self.target_col}_rollstd_{days}d'] = target.rolling(window, min_periods=1).std()
            
        # ----- Safe Percentage Change -----
        X_new[f'{self.target_col}_pct_change'] = target.pct_change().replace([np.inf, -np.inf], np.nan) #infine values need to be replaced other wise throws error
        
        # ----- Wind Direction needs fixing other wise throws error -
        if 'wd_(deg)' in X.columns:
            # have to convert to radians first
            wd_rad = np.deg2rad(X_new['wd_(deg)'])
            X_new['wd_sin'] = np.sin(wd_rad)
            X_new['wd_cos'] = np.cos(wd_rad)
            X_new = X_new.drop('wd_(deg)', axis=1)
            
        #remove null values
        X_new = X_new.replace([np.inf, -np.inf], np.nan)
        X_new = X_new.fillna(method='ffill').fillna(method='bfill')
        
        return X_new

def xy_splitter(train, validation, test):  
    """
    Splits the training, validation, and test DataFrames into feature and target sets.

    Params:
    train (pd.DataFrame): The training DataFrame.
    validation (pd.DataFrame): The validation DataFrame.
    test (pd.DataFrame): The test DataFrame.

    Returns:
    X (pd.DataFrame): The feature set for the provided dataset.
    y (pd.Series): The target set for the provided dataset.
    """  

    target = 'T_(degC)'
    X_train = train.drop(target, axis=1)
    y_train = train[target]

    X_test = test.drop(target, axis=1)
    y_test = test[target]

    X_val = validation.drop(target, axis=1)
    y_val = validation[target]

    train_dropped = len(X_train) - len(train)
    val_dropped = len(X_val) - len(validation)
    test_dropped = len(X_test) - len(test)
    
    return (
        (X_train, y_train),
        (X_val, y_val),
        (X_test, y_test),
        {
            'dropped': {
                'train': train_dropped,
                'val': val_dropped,
                'test': test_dropped}}
            )