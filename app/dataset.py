import yfinance as yf
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

class DatasetManager:
    def __init__(self, ticker: str, start_date: str, end_date: str=None):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.scaler = None
        self.close_scaler = None
        self.close_scaler = None

    def download_data(self):
        data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        return data
    
    def split_data(self, df, features, train_pct, lookback, device):
        data = []

        for i in range(len(df) - lookback):
            data.append(df[features][i:i+lookback])

        data = np.array(data)

        train_size = int(train_pct*len(data))

        x_train = torch.from_numpy(data[:train_size, :-1, :]).type(torch.Tensor).to(device)
        y_train = torch.from_numpy(data[:train_size, -1, :]).type(torch.Tensor).to(device)

        x_test = torch.from_numpy(data[train_size:, :-1, :]).type(torch.Tensor).to(device)
        y_test = torch.from_numpy(data[train_size:, -1, :]).type(torch.Tensor).to(device)

        return x_train, y_train, x_test, y_test
    
    def invert_transform_data(self, y_train_pred=None, y_train=None, y_test_pred=None, y_test=None):
        if y_train_pred is not None:
            y_train_pred = self.close_scaler.inverse_transform(y_train_pred.detach().cpu().numpy())
        if y_train is not None:
            y_train = self.close_scaler.inverse_transform(y_train.detach().cpu().numpy())
        if y_test_pred is not None:
            y_test_pred = self.close_scaler.inverse_transform(y_test_pred.detach().cpu().numpy())
        if y_test is not None:
            y_test = self.close_scaler.inverse_transform(y_test.detach().cpu().numpy())
        return y_train_pred, y_train, y_test_pred, y_test
    
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
    
    def remove_features(self, df, features_to_remove):
        df = df.drop(columns=features_to_remove)
        return df
    
    def normalize_data(self, df):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_array = self.scaler.fit_transform(df)
        scaled_df = pd.DataFrame(scaled_array, columns=df.columns, index=df.index)

        self.close_scaler = MinMaxScaler(feature_range=(0, 1))
        self.close_scaler.fit(df[['Close']])

        scaled_df = scaled_df.dropna()
        return scaled_df
    
    def get_features(self, df):
        feature_list = df.columns.tolist()
        features = [feature[0] for feature in feature_list if len(feature)>1]
        return features
    
