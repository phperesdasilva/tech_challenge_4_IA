import yfinance as yf
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from datetime import date, timedelta
 

class DatasetManager:
    def __init__(self, ticker: str, start_date: str, end_date: str=None):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def download_data(self):
        data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        return data
    
    def split_data(self, df, features, train_pct):
        
        split = int(train_pct*len(df[features]))
        df_train = df.iloc[:split]
        df_test = df.iloc[split:]
        y_train = df_train[features].values
        y_test = df_test[features].values

        return df_train, y_train, df_test, y_test
    
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
    
    def normalize_data(self, y_train, y_test):
        self.scaler.fit(y_train.reshape(-1, 1))
        y_train_scaled = self.scaler.transform(y_train.reshape(-1, 1)).flatten()
        y_test_scaled = self.scaler.transform(y_test.reshape(-1, 1)).flatten()
        return y_train_scaled, y_test_scaled
    
    def get_features(self, df):
        feature_list = df.columns.tolist()
        features = [feature[0] for feature in feature_list if len(feature)>1]
        return features

    def get_api_tensor(self, lookback: int, device: torch.device = torch.device('cpu')) -> torch.Tensor:
        df = self.download_data()
        features_to_remove = ['Open', 'High', 'Low', 'Volume']
        df = self.remove_features(df, features_to_remove)
        df = df.dropna()
        scaled_df = self.normalize_data(df)
        window = scaled_df.tail(lookback)
        if len(window) < lookback:
            raise ValueError('Janela de dados insuficiente para montar o tensor de entrada.')

        arr = window.to_numpy(dtype=np.float32)
        # Ensure the NumPy array is writable and contiguous before converting to a torch tensor
        if not arr.flags.writeable:
            arr = arr.copy()
        arr = np.ascontiguousarray(arr)
        tensor = torch.from_numpy(arr).unsqueeze(0).to(device)
        return tensor

    
