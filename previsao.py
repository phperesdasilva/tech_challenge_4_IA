import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import plotly.express as px
import plotly.io as pio
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
import yfinance as yf
from curl_cffi import requests
import os

torch.manual_seed(42)
np.random.seed(42)

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=80, num_layers=2, output_size=15):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def gerar_previsao(ticker: str, periodo_anos: int, salvar_modelo: bool = True):
    """
    Treina LSTM e retorna gráfico + métricas.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    # ── Dados via yfinance ──
    session = requests.Session(impersonate="chrome")
    stock = yf.Ticker(f'{ticker.upper()}.SA', session=session)
    data = stock.history(period=f'{periodo_anos}y')

    if data.empty:
        return {"erro": f"Dados não encontrados para {ticker}"}

    df = data.reset_index()[['Date', 'Close']]

    # ── Separar treino/teste (80/20) ──
    split = int(0.8 * len(df))
    df_train, df_teste = df.iloc[:split], df.iloc[split:]
    y_train, y_test = df_train['Close'].values, df_teste['Close'].values

    # ── Normalizar ──
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(y_train.reshape(-1, 1))
    y_train_s = scaler.transform(y_train.reshape(-1, 1)).flatten()
    y_test_s = scaler.transform(y_test.reshape(-1, 1)).flatten()

    # ── Sequências LSTM ──
    lookback, horizonte = 60, 15
    X_tr, y_tr = [], []
    for i in range(len(y_train_s) - lookback - horizonte + 1):
        X_tr.append(y_train_s[i:i + lookback])
        y_tr.append(y_train_s[i + lookback:i + lookback + horizonte])
    X_te, y_te = [], []
    for i in range(len(y_test_s) - lookback - horizonte + 1):
        X_te.append(y_test_s[i:i + lookback])
        y_te.append(y_test_s[i + lookback:i + lookback + horizonte])

    X_train = torch.FloatTensor(np.array(X_tr)).reshape(-1, 60, 1)
    y_train = torch.FloatTensor(np.array(y_tr))
    X_test = torch.FloatTensor(np.array(X_te)).reshape(-1, 60, 1)
    y_test_t = torch.FloatTensor(np.array(y_te))

    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

    model = LSTM()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for epoch in range(200):
        loss_avg = 0
        for Xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(Xb), yb)
            loss.backward()
            opt.step()
            loss_avg += loss.item()
        if (epoch + 1) % 50 == 0:
            print(f'  Época {epoch+1}/200 — Loss: {loss_avg/len(loader):.6f}')

    # ── Predição ──
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).numpy()

    y_train_real = scaler.inverse_transform(y_train.numpy().reshape(-1, 1)).reshape(-1, 15)
    y_test_real = scaler.inverse_transform(y_test_t.numpy().reshape(-1, 1)).reshape(-1, 15)
    y_pred_real = scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(-1, 15)

    mape = mean_absolute_percentage_error(y_test_real, y_pred_real)

    # ── Salvar modelo ──
    caminho = None
    if salvar_modelo:
        nome = f'modelo_lstm_{ticker.lower()}.pth'
        caminho = os.path.join(os.getcwd(), nome)
        torch.save(model.state_dict(), caminho)

    # ── Montar DataFrame para o gráfico ──
    train_end = split - horizonte + 1
    test_start = split + lookback

    df_plot = pd.concat([
        pd.DataFrame({'Date': df['Date'].iloc[lookback:train_end],
                      'Price': y_train_real[:, 0],
                      'Tipo': 'Treino'}),
        pd.DataFrame({'Date': df['Date'].iloc[split:split + lookback],
                      'Price': df['Close'].iloc[split:split + lookback].values,
                      'Tipo': 'Real (Teste)'}),
        pd.DataFrame({'Date': df['Date'].iloc[test_start:test_start + len(y_pred_real)],
                      'Price': y_test_real[:, 0],
                      'Tipo': 'Real (Teste)'}),
        pd.DataFrame({'Date': df['Date'].iloc[test_start:test_start + len(y_pred_real)],
                      'Price': y_pred_real[:, 0],
                      'Tipo': 'Previsto'}),
    ], ignore_index=True)

    # ── Gráfico Plotly ──
    fig = px.line(df_plot, x='Date', y='Price', color='Tipo',
                  title=f'🧠 LSTM — Previsão {ticker.upper()} ({periodo_anos} ano(s)) | MAPE: {mape:.2%}',
                  labels={'Price': 'Preço (R$)', 'Date': 'Data', 'Tipo': 'Tipo'})
    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#ffffff'),
                      hovermode='x unified', height=600)

    return {"grafico": pio.to_html(fig, include_plotlyjs='cdn', full_html=False),
            "mape": mape, "modelo": caminho}


def previsao_futuro(ticker: str, periodo_anos: int = 5):
    """
    Carrega modelo LSTM salvo e prevê os próximos 15 dias.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    session = requests.Session(impersonate="chrome")
    stock = yf.Ticker(f'{ticker.upper()}.SA', session=session)
    data = stock.history(period=f'{periodo_anos}y')

    if data.empty:
        return {"erro": f"Dados não encontrados para {ticker}"}

    close_prices = data['Close'].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(close_prices)

    seq_len = 60
    scaled_data = scaler.transform(close_prices)
    last_sequence = scaled_data[-seq_len:].reshape(1, seq_len, 1)

    # ── Carregar modelo salvo ──
    caminho_modelo = f'modelo_lstm_{ticker.lower()}.pth'
    try:
        model = LSTM(input_size=1, hidden_size=80, num_layers=2, output_size=15)  # ← agora funciona!
        model.load_state_dict(torch.load(caminho_modelo, map_location='cpu'))
        model.eval()
    except FileNotFoundError:
        return {"erro": f"Modelo não encontrado. Execute o treinamento primeiro: {caminho_modelo}"}

    seq_tensor = torch.FloatTensor(last_sequence)
    with torch.no_grad():
        pred = model(seq_tensor)

    predictions = pred.cpu().detach().numpy()[0]
    valores_previstos = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

    from datetime import datetime, timedelta
    ultima_data = data.index[-1]
    dias_uteis = []
    data_atual = ultima_data + timedelta(days=1)
    while len(dias_uteis) < 15:
        if data_atual.weekday() < 5:
            dias_uteis.append(data_atual)
        data_atual += timedelta(days=1)

    previsoes = []
    for i in range(15):
        previsoes.append({
            "dia": i + 1,
            "data": dias_uteis[i].strftime('%d/%m/%Y'),
            "valor": round(float(valores_previstos[i]), 2)
        })

    return {
        "ticker": ticker.upper(),
        "ultimo_preco": round(float(close_prices[-1][0]), 2),
        "ultima_data": ultima_data.strftime('%d/%m/%Y'),
        "previsoes": previsoes
    }
