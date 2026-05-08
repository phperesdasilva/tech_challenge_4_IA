import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import torch

class DatasetManager:
    def __init__(self, ticker: str, years: int):
        self.ticker = ticker
        self.years = years
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def download_data(self):
        df = yf.download(self.ticker, period=f'{self.years}y')
        data = pd.DataFrame(df).reset_index()
        data['Date'] = data['Date'].astype(str)
        return data
    
    def split_data(self, df, test_feature, train_pct):
        
        split = int(train_pct*len(df[test_feature]))
        df_train = df.iloc[:split]
        df_test = df.iloc[split:]
        y_train = df_train[test_feature].values
        y_test = df_test[test_feature].values

        return df_train, y_train, df_test, y_test

    def preprocess_data(self, df):
        df['7_day_MA'] = df['Close'].rolling(window=7).mean()  
        df['21_day_MA'] = df['Close'].rolling(window=21).mean()  
        df['Daily_Return'] = df['Close'].pct_change()
        df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))
        df["Volatility_7"] = df["Daily_Return"].rolling(7).std()
        df["Volatility_21"] = df["Daily_Return"].rolling(21).std()
        df["Volume_MA_10"] = df["Volume"].rolling(10).mean()
        df["Volume"] = df["Volume"].astype(float)
        df.dropna()
        return df
    
    def filter_features(self, df, features_to_keep):
        df = df[features_to_keep]
        return df
    
    def normalize_data(self, y_train, y_test):
        self.scaler.fit(y_train.reshape(-1, 1))
        y_train_scaled = self.scaler.transform(y_train.reshape(-1, 1)).flatten()
        y_test_scaled = self.scaler.transform(y_test.reshape(-1, 1)).flatten()
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

        x_train = np.array(x_train_list).reshape(-1, 60, 1)
        y_train = np.array(y_train_list)
        x_test = np.array(x_test_list).reshape(-1, 60, 1)
        y_test = np.array(y_test_list)

        x_train_tensor = torch.FloatTensor(x_train)
        y_train_tensor = torch.FloatTensor(y_train)
        x_test_tensor = torch.FloatTensor(x_test)
        y_test_tensor = torch.FloatTensor(y_test)

        return x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor