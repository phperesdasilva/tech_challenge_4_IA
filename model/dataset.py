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
    """Gerencia download, pré-processamento e preparação dos dados.

    Args:
        tickers (list[str] | str): Símbolo(s) de ações para download via yfinance.
        years (int): Quantidade de anos de histórico a baixar.
    """
    def __init__(self, tickers: list[str], years: int):
        self.tickers = tickers
        self.years = years
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def download_data(self):
        """Faz o download dos dados históricos via yfinance.

        Returns:
            pandas.DataFrame: Dados históricos com a coluna `Date` como string.
        """
        df = yf.download(self.tickers, period=f'{self.years}y')
        data = pd.DataFrame(df).reset_index()
        data['Date'] = data['Date'].astype(str)
        return data
    
    def split_data(self, df, test_feature, train_pct):
        """Divide o DataFrame em conjuntos de treino e teste.

        Args:
            df (DataFrame): DataFrame com os dados.
            test_feature (str): Nome da coluna alvo (ex.: 'Close').
            train_pct (float): Proporção de dados usados para treino (0-1).

        Returns:
            tuple: `(df_train, y_train, df_test, y_test, split_index)`.
        """
        split = int(train_pct*len(df[test_feature]))
        df_train = df.iloc[:split]
        df_test = df.iloc[split:]
        y_train = df_train[test_feature].values
        y_test = df_test[test_feature].values

        return df_train, y_train, df_test, y_test, split
    
    def filter_features(self, df, features_to_keep):
        """Filtra o DataFrame mantendo apenas as colunas especificadas.

        Args:
            df (DataFrame): DataFrame de entrada.
            features_to_keep (list[str]): Lista de colunas a manter.

        Returns:
            DataFrame: DataFrame filtrado.
        """
        df = df[features_to_keep]
        return df
    
    def normalize_data(self, y_train, y_test):
        """Normaliza `y_train` e `y_test` usando MinMaxScaler e salva o scaler.

        O scaler é treinado a partir de `y_train` e aplicado a ambos os vetores.

        Args:
            y_train (np.ndarray): Valores de treino da série alvo.
            y_test (np.ndarray): Valores de teste da série alvo.

        Returns:
            tuple: `(y_train_scaled, y_test_scaled)` como arrays 1D.
        """
        self.scaler.fit(y_train.reshape(-1, 1))
        y_train_scaled = self.scaler.transform(y_train.reshape(-1, 1)).flatten()
        y_test_scaled = self.scaler.transform(y_test.reshape(-1, 1)).flatten()
        
        if '.' in self.tickers:
            ticker = self.tickers.replace('.', '_')
        else:
            ticker = self.tickers

        scaler_file = f'scaler_files/{ticker}_{params['scaler_path']}'
        joblib.dump(self.scaler, scaler_file)
        return y_train_scaled, y_test_scaled
    
    def get_features(self, df):
        """Retorna a lista de features presentes no DataFrame multi-index.

        Trata casos em que `df.columns` esteja em formato multi-index e extrai
        apenas os nomes relevantes das features.
        """
        feature_list = df.columns.tolist()
        features = [feature[0] for feature in feature_list if len(feature)>1]
        return features

    def create_train_test_sequences(self, lookback_period, prediction_period, y_train_scaled, y_test_scaled):
        """Gera sequências de entrada e saída para treino e teste.

        Args:
            lookback_period (int): Número de passos de histórico usados como entrada.
            prediction_period (int): Número de passos a prever.
            y_train_scaled (np.ndarray): Série de treino escalada.
            y_test_scaled (np.ndarray): Série de teste escalada.

        Returns:
            tuple: `(x_train, y_train, x_test, y_test)` prontos para redes LSTM.
        """
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
        """Aplica a transformação inversa do scaler aos dados fornecidos.

        Args:
            data (np.ndarray): Dados escalados.

        Returns:
            np.ndarray: Dados na escala original.
        """
        return self.scaler.inverse_transform(data.reshape(-1, 1)).reshape(data.shape)