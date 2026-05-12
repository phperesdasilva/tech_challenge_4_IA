import torch
import yfinance as yf
from flask import Flask, request, jsonify
from flasgger import Swagger
import mlflow
from global_params import params
import joblib

app = Flask(__name__)

swagger = Swagger(app, template={
    "info": {
        "title": "Stock Prediction API",
        "description": "API para previsão de preços de ações usando LSTM (Tech Challenge 4)",
        "version": "1.0.0"
    }
})

with open("last_run_id.txt", "r", encoding="utf-8") as arquivo:
    run_id = arquivo.read()

MODEL_URI = f"runs:/{run_id}/{params['model_name']}"
print(f'\n\nModel URI: {MODEL_URI}\n\n')

mlflow.set_tracking_uri(params['mlflow_tracking_uri'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_resources():
    model = mlflow.pytorch.load_model(MODEL_URI, map_location=device)
    scaler = joblib.load(params['scaler_path'])
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

        data = yf.download(ticker, period="5y")
        if data.empty:
            return jsonify({"error": "Ticker não encontrado ou sem dados"}), 404

        df_input = data[['Close']].tail(params['lookback_period']).values
        
        scaled_data = scaler.transform(df_input)
        
        input_tensor = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = model(input_tensor)
            predictions_scaled = y_pred.cpu().numpy()

        inverse_predictions = scaler.inverse_transform(predictions_scaled.reshape(-1, 1))

        return jsonify({
            "ticker": ticker,
            "predictions": inverse_predictions.flatten().tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)