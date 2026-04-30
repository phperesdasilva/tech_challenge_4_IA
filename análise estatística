import yfinance as yf
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from curl_cffi import requests
from statsmodels.graphics.tsaplots import month_plot, quarter_plot, plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf

#Inserir ticker:
ticker = input(f'Insira o ticker desejado:')
periodo = input(f'Insira o período desejado em anos (mínimo 1 ano - ex 1):')

#Baixar dados do YahooFinance:
session = requests.Session(impersonate="chrome")
stock = yf.Ticker(f'{ticker.upper()}.SA', session=session)
data = stock.history(period=f'{periodo}y')
data.to_csv(f'{ticker.lower()}_historico_{periodo}anos.csv')
data2 = pd.DataFrame(data)
data2 = data2[['Close']]

#Tratamento dos dados:
data2 = data2.reset_index()
data2['Date'] = pd.to_datetime(data2['Date'])
data2 = data2.rename(columns={'Date':'ds','Close':'y'})
data2.set_index('ds',inplace=True)
data2['y'] = data2['y'].interpolate(method='linear',limit_direction='both')
#data2.index = data2.index.strftime('%Y-%m-%d')

#Análise estatística:
#Gŕafico cotação histórica no período desejado
cotacao = px.line(data2.reset_index(),x='ds',y='y',title=f'Cotação {ticker} período {periodo} anos',labels={'ds': 'Período', 'y': 'Preço (R$)'})
cotacao.show()

#Gŕafico decomposição sazonal
decompose = seasonal_decompose(data2['y'],model='multiplicative',period=365)
# Criar subplots
fig = make_subplots(
    rows=4, cols=1,
    subplot_titles=('Série Original', 'Tendência', 'Sazonal', 'Resíduo'),
    vertical_spacing=0.08
)

# Adicionar todos os gráficos
fig.add_trace(
    go.Scatter(x=data2.index, y=data2['y'], name='Original', line=dict(color='blue')),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=decompose.trend.index, y=decompose.trend, name='Tendência', line=dict(color='green')),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(x=decompose.seasonal.index, y=decompose.seasonal, name='Sazonal', line=dict(color='orange')),
    row=3, col=1
)

fig.add_trace(
    go.Scatter(x=decompose.resid.index, y=decompose.resid, name='Resíduo', line=dict(color='red')),
    row=4, col=1
)

# Configurar layout
fig.update_layout(
    height=900,
    title_text=f'Decomposição Sazonal - {ticker} ({periodo} anos)',
    showlegend=False
)

fig.show()

# Preparar dados
monthly = data2['y'].resample('ME').mean()
quarterly = data2['y'].resample('QE').mean()
acf_vals = acf(data2['y'], nlags=100)
pacf_vals = pacf(data2['y'], nlags=100)

# Criar mosaico 2x2
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Monthly Plot', 'Quarterly Plot', 'ACF', 'PACF'),
    vertical_spacing=0.12,
    horizontal_spacing=0.1
)

# Month plot
fig.add_trace(
    go.Scatter(x=monthly.index, y=monthly.values, mode='lines+markers', name='Monthly'),
    row=1, col=1
)

# Quarter plot
fig.add_trace(
    go.Scatter(x=quarterly.index, y=quarterly.values, mode='lines+markers', name='Quarterly'),
    row=1, col=2
)

# ACF
fig.add_trace(
    go.Scatter(x=list(range(len(acf_vals))), y=acf_vals, mode='lines+markers', name='ACF'),
    row=2, col=1
)

# PACF
fig.add_trace(
    go.Scatter(x=list(range(len(pacf_vals))), y=pacf_vals, mode='lines+markers', name='PACF'),
    row=2, col=2
)

# Adicionar linha zero em ACF e PACF
fig.add_hline(y=0, line_dash='dash', line_color='black', row=2, col=1)
fig.add_hline(y=0, line_dash='dash', line_color='black', row=2, col=2)

# Configurar layout
fig.update_layout(
    height=800,
    title_text=f'Análise de Série Temporal - {ticker}',
    showlegend=False
)

# Adicionar títulos dos eixos
fig.update_xaxes(title_text='Data', row=1, col=1)
fig.update_xaxes(title_text='Data', row=1, col=2)
fig.update_xaxes(title_text='Lag', row=2, col=1)
fig.update_xaxes(title_text='Lag', row=2, col=2)

fig.update_yaxes(title_text='Preço', row=1, col=1)
fig.update_yaxes(title_text='Preço', row=1, col=2)
fig.update_yaxes(title_text='ACF', row=2, col=1)
fig.update_yaxes(title_text='PACF', row=2, col=2)

fig.show()
