from pathlib import Path
import sys

import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from api.global_params import params
import pandas as pd
import torch
import joblib

torch.manual_seed(42)
np.random.seed(42)

class DatasetManager:
    def __init__(self, tickers: list[str], years: int):
        self.tickers = tickers
        self.years = years
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def download_data(self):
        df = yf.download(self.tickers, period=f'{self.years}y')
        data = pd.DataFrame(df).reset_index()
        data['Date'] = data['Date'].astype(str)
        return data
    
    def split_data(self, df, test_feature, train_pct):
        
        split = int(train_pct*len(df[test_feature]))
        df_train = df.iloc[:split]
        df_test = df.iloc[split:]
        y_train = df_train[test_feature].values
        y_test = df_test[test_feature].values

        return df_train, y_train, df_test, y_test, split
    
    def filter_features(self, df, features_to_keep):
        df = df[features_to_keep]
        return df
    
    def normalize_data(self, y_train, y_test):
        self.scaler.fit(y_train.reshape(-1, 1))
        y_train_scaled = self.scaler.transform(y_train.reshape(-1, 1)).flatten()
        y_test_scaled = self.scaler.transform(y_test.reshape(-1, 1)).flatten()
        
        if '.' in self.tickers:
            ticker = self.tickers.replace('.', '_')

        scales_file = f'scales_files/{ticker}_{params['scaler_path']}'
        joblib.dump(self.scaler, params['scaler_path'])
        return y_train_scaled, y_test_scaled
    
    def get_features(self, df):
        feature_list = df.columns.tolist()
        features = [feature[0] for feature in feature_list if len(feature)>1]
        return features

    def create_train_test_sequences(self, lookback_period, prediction_period, y_train_scaled, y_test_scaled):
        
        x_train_list = []
        y_train_list = []
        for i in range(len(y_train_scaled)-lookback_period-prediction_period+1):
            x_train_list.append(y_train_scaled[i:(i+lookback_period)])
            y_train_list.append(y_train_scaled[i+lookback_period:i+lookback_period+prediction_period])

        x_test_list = []
        y_test_list = []
        for i in range(len(y_test_scaled)-lookback_period-prediction_period+1):
            x_test_list.append(y_test_scaled[i:(i+lookback_period)])
            y_test_list.append(y_test_scaled[i+lookback_period:i+lookback_period+prediction_period])

        x_train = np.array(x_train_list).reshape(-1, params['lookback_period'], 1)
        y_train = np.array(y_train_list)
        x_test = np.array(x_test_list).reshape(-1, params['lookback_period'], 1)
        y_test = np.array(y_test_list)

        return x_train, y_train, x_test, y_test
    
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data.reshape(-1, 1)).reshape(data.shape)