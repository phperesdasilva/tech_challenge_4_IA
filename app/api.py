import flask
from flasgger import Swagger

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
    value = 42
    return {'value': value, 'type': type(value).__name__}, 200


if __name__ == '__main__':
    app.run(debug=True, port=5000)