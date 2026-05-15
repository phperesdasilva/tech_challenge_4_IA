import os

import requests
import matplotlib.pyplot as plt

# Configurações da API
API_URL = "http://localhost:5000/predict"
TICKER = "PETR4.SA"

def test_prediction():
    payload = {"ticker": TICKER}
    
    print(f"Enviando requisição para {TICKER}...")
    
    try:
        response = requests.post(API_URL, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            predictions = data['predictions']
            
            print("\n✅ Sucesso!")
            print(f"Ticker: {data['ticker']}")
            print(f"Previsões para os próximos 15 dias:")
            for i, price in enumerate(predictions, 1):
                print(f"Dia {i:02d}: R$ {price:.2f}")
            
            # Gerar gráfico das previsões
            plot_results(predictions, data['ticker'])
            
        else:
            print(f"\n❌ Erro {response.status_code}: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("\n❌ Erro: Não foi possível conectar à API. Verifique se o Flask está rodando em http://localhost:5000")

def plot_results(preds, ticker):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 16), preds, marker='o', linestyle='-', color='b', label='Previsão')
    plt.title(f"Previsão de Preço - Próximos 15 Dias ({ticker})")
    plt.xlabel("Dias Futuros")
    plt.ylabel("Preço de Fechamento (R$)")
    plt.xticks(range(1, 16))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()

def should_retrain_model(ticker):
    for model in os.listdir('model_files'):
        file_ticker = model.split('_')[0]
        real_ticker = ticker.split('.')[0]

        if file_ticker == real_ticker:
            print('Model found in model_files')
            return False
        
    print('Model not found in model_files')
    return True  


if __name__ == "__main__":
    # test_prediction()
    from model.model_trainer import train_model
    train_model('PETR4.SA')
    print(should_retrain_model('PETR4.SA'))