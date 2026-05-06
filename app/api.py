from datetime import date, timedelta
import flask
from flasgger import Swagger
import numpy as np
import torch
import yfinance as yf
from lstm import StockLSTM
from dataset import DatasetManager

ticker = 'MSFT'

device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = 1
hidden_size = 32
num_layers = 2
prediction_period = 1
dropout = 0.2

lookback_period = 30

pth_path = "trained_model.pth"

app = flask.Flask(__name__)
swagger = Swagger(app)


@app.route('/health', methods=['GET'])
def health():
    """
    Health Check endpoint
    ---
    tags:
      - Health
    responses:
      200:
        description: API is healthy
        schema:
          properties:
            status:
              type: string
              example: healthy
            message:
              type: string
              example: API is running
      500:
        description: API is unhealthy
    """
    try:
        return {'status': 'healthy', 'message': 'API is running'}, 200
    except Exception as e:
        return {'status': 'unhealthy', 'message': f'API error: {str(e)}'}, 500

@app.route('/prediction', methods=['GET'])
def prediction():
    """
    Prediction endpoint
    ---
    tags:
      - Prediction
    responses:
      200:
        description: Returns a prediction value
        schema:
          properties:
            value:
              type: integer
              example: 42
            type:
              type: string
              example: int
    """
    try:
        model = StockLSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            output_size=prediction_period, #número de dias a serem previstos
            dropout=dropout).to(device)
        
        model.load_state_dict(torch.load(pth_path, map_location=device))
        model.eval()

        lookback_date = str(date.today() - timedelta(days=60))

        dataset = DatasetManager(ticker, start_date=lookback_date)
        input_tensor = dataset.get_api_tensor(lookback=lookback_period, device=device)

        with torch.no_grad():
            output = model(input_tensor)

        value = float(output.item())
        return {'value': value, 'type': type(output).__name__}, 200

    except Exception as e:
        return {'status': 'error', 'message': f'Prediction error: {str(e)}'}, 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)