import torch
import numpy as np
import yfinance as yf
from flask import Flask, request, jsonify
from flasgger import Swagger
from lstm import StockLSTM
from global_params import params
import joblib

app = Flask(__name__)
# Inicializa o Swagger com configurações básicas
swagger = Swagger(app, template={
    "info": {
        "title": "Stock Prediction API",
        "description": "API para previsão de preços de ações usando LSTM (Tech Challenge 4)",
        "version": "1.0.0"
    }
})

# Configurações para Modelo Univariado
MODEL_PATH = params['model_path']
SCALER_PATH = params['scaler_path']
SEQ_LENGTH = params['lookback_period']
INPUT_SIZE = 1
HIDDEN_SIZE = params['hidden_size']
PREDICTION_DAYS = params['prediction_period']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_resources():
    # Instancia o modelo com a saída correta (15) para evitar size mismatch
    model = StockLSTM(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=PREDICTION_DAYS)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_resources()

@app.route('/health', methods=['GET'])
def health():
    """
    Endpoint de verificação de saúde da API.
    ---
    responses:
      200:
        description: API está online e recursos carregados.
        schema:
          properties:
            status:
              type: string
              example: "healthy"
            device:
              type: string
              example: "cuda"
            model_loaded:
              type: boolean
            scaler_loaded:
              type: boolean
    """
    health_status = {
        "status": "healthy",
        "device": str(device),
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }
    
    # Se algum recurso crítico não estiver carregado, retorna erro 503
    if not model or not scaler:
        health_status["status"] = "unhealthy"
        return jsonify(health_status), 503
        
    return jsonify(health_status), 200
    

@app.route('/predict', methods=['POST'])
def predict_next_days():
    """
    Realiza a previsão do preço de fechamento para os próximos 15 dias.
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - ticker
          properties:
            ticker:
              type: string
              example: "PETR4.SA"
              description: O símbolo da ação no Yahoo Finance.
    responses:
      200:
        description: Previsões geradas com sucesso.
        schema:
          properties:
            ticker:
              type: string
            predictions:
              type: array
              items:
                type: number
              description: Lista com os preços previstos para os próximos 15 dias.
      400:
        description: Erro na requisição (ex. ticker faltando).
      404:
        description: Ticker não encontrado no Yahoo Finance.
      500:
        description: Erro interno no processamento ou carregamento do modelo.
    """
    try:
        content = request.json
        ticker = content.get('ticker')
        
        if not ticker:
            return jsonify({"error": "Ticker não fornecido"}), 400

        # 1. Busca dados no yfinance
        data = yf.download(ticker, period="5y")
        if data.empty:
            return jsonify({"error": "Ticker não encontrado ou sem dados"}), 404

        # Pega as últimas 60 janelas de fechamento
        df_input = data[['Close']].tail(SEQ_LENGTH).values
        
        # 2. Escalonamento
        scaled_data = scaler.transform(df_input)
        
        # 3. Previsão Direta (O modelo já cospe 15 dias)
        input_tensor = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = model(input_tensor) # Shape esperado: [1, 15]
            predictions_scaled = y_pred.cpu().numpy()

        # 4. Inversão do Scaler
        # Como o scaler foi treinado para 1 feature, precisamos dar reshape para (15, 1)
        inverse_predictions = scaler.inverse_transform(predictions_scaled.reshape(-1, 1))

        return jsonify({
            "ticker": ticker,
            "predictions": inverse_predictions.flatten().tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)