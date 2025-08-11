''' build random forest ensemble model. Hyperparameter tuning '''
import pandas as pd
train = pd.read_csv('train.csv')

train['transaction_year'] = train['transaction_year_month'].astype(str).str[:4].astype(int)
train['transaction_month'] = train['transaction_year_month'].astype(str).str[4:].astype(int)

