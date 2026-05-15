# tech_challenge_4_IA

**Visão geral**
- **Descrição**: Projeto de previsão de preços de ações usando um modelo LSTM em PyTorch. Fornece uma API Flask que aceita um `ticker` e retorna previsões para os próximos dias. Inclui scripts para treinar modelos, salvar pesos e escalers, e registrar execuções no MLflow.

**Estrutura do repositório**
- **api**: Aplicação Flask, rotas e templates (página web).
- **model**: Código do dataset, arquitetura LSTM e rotina de treinamento.
- **model_files / scaler_files**: Diretórios onde pesos do modelo e scalers são armazenados após o treinamento.
- **api_test.py**: Script de teste rápido da API.

**Arquitetura do projeto**
- **API (`api`)**: Aplicação Flask que expõe endpoints para saúde (`/health`), página web (`/`) e predição (`/predict`). A API carrega modelos e scalers a partir de `model_files` e `scaler_files` e pode disparar o re-treinamento via `train_model` quando necessário.
- **Módulo de modelo (`model`)**: Contém a classe `DatasetManager` (download e pré-processamento), `StockLSTM` (arquitetura do modelo) e `model_trainer.py` (rotina de treinamento, logging MLflow, salvamento de artefatos).
- **Armazenamento de artefatos**: `model_files/` (pesos do modelo) e `scaler_files/` (scalers por ticker). Arquivos auxiliares `last_run_id.txt` e `last_run_date.txt` rastreiam execuções.

**Ingestão de dados**
- Fonte: dados históricos baixados via `yfinance` (ticker(s) e período configuráveis).
- Passos do pipeline de ingestão (implementado em `model/dataset.py`):
	1. `DatasetManager.download_data()`: baixa preços ajustados para os tickers solicitados e normaliza a coluna `Date` como string.
	2. `filter_features()`: seleciona somente as colunas necessárias (por exemplo `Close`).
	3. `split_data()`: divide em treino/teste com proporção configurável (ex.: 80/20).
	4. `normalize_data()`: treina um `MinMaxScaler` no conjunto de treino, aplica aos conjuntos e salva o scaler em `scaler_files/{TICKER}_{lstm_scaler.pkl}`.
	5. `create_train_test_sequences()`: gera janelas/seqüências para entrada (`lookback_period`) e targets (`prediction_period`) prontas para o treinamento LSTM.

**Treinamento e registro**
- O treinamento é realizado por `train_model(ticker)` em `model/model_trainer.py`.
- O fluxo de treinamento cria tensores PyTorch, treina com `DataLoader`, calcula métricas (MAPE, MAE, RMSE) e registra parâmetros/metrics no MLflow.
- Após o treino o modelo é salvo em `model_files/{TICKER}_{lstm_model_weights.pt2}` e registrado no MLflow com `mlflow.pytorch.log_model`.

**Como testar o projeto**

1) Testar com Docker
- Pré-requisito: Docker e Docker Compose.
- Build + up:

```bash
docker compose build
docker compose up
```

- A API estará disponível em `http://0.0.0.0:5000` por padrão (ver `api.app`), e o MLflow UI pode estar exposto conforme `docker-compose.yml` (ex.: `http://localhost:3050`).

2) Testar usando `api_test.py`
- Pré-requisito: ambiente virtual com dependências instaladas (`pip install -r requirements.txt`).
- Executar:

```bash
source venv/bin/activate
python api_test.py
```

- O script `api_test.py` faz chamadas de integração simples contra a API local. Use-o para validar endpoints sem subir contêineres.

3) Testar por linha de comando (sem `api_test.py`)
- Executar localmente a API:

```bash
source venv/bin/activate
python -m api.app
```

- Fazer uma requisição de predição via `curl` (substitua `PETR4.SA` pelo ticker desejado):

```bash
curl -X POST http://localhost:5000/predict \
	-H "Content-Type: application/json" \
	-d '{"ticker": "PETR4.SA"}'
```

- Treinar manualmente um modelo via linha de comando (exemplo rápido):

```bash
python -c "from model.model_trainer import train_model; train_model('PETR4.SA')"
```

**Dicas e pontos a validar**
- Ao iniciar a API sem modelos salvos, a chamada `/predict` pode acionar o treinamento, que é custoso (CPU/GPU e tempo). Use um ticker pequeno para testes rápidos.
- Verifique `MLFLOW_TRACKING_URI` se quiser ver as execuções no UI do MLflow.
- Se for rodar em produção, considere: persistência de artefatos em armazenamento externo, enfileiramento de treinos (ex.: Celery), e endpoints assíncronos para evitar bloqueio durante o re-treinamento.

---
Atualizado em: 2026-05-15

**Dependências**
- **Python 3.8+**: Ambiente virtual recomendado.
- Pacotes principais: `flask`, `torch`, `yfinance`, `scikit-learn`, `mlflow`, `joblib`, `flasgger`, `flask-cors`.
- Instale com:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Configuração e execução**
- Defina (opcional) a URI do MLflow via variável de ambiente `MLFLOW_TRACKING_URI`.
- Inicie a API localmente:

```bash
export MLFLOW_TRACKING_URI=http://localhost:3050
python -m api.app
```

Ou via Docker Compose (se houver configuração):

```bash
docker compose up --build
```

**Endpoints principais**
- **GET /**: Página HTML com formulário de previsão.
- **GET /health**: Retorna status da aplicação e informações básicas.
- **POST /predict**: Recebe JSON com `{"ticker": "SYM"}` e retorna previsões (lista de valores). Se o modelo não existir localmente, a API tenta treinar e retorna `model_retrained` e `run_id`.

**Treinamento do modelo**
- Chamando `train_model(ticker)` dentro de `model/model_trainer.py` o pipeline fará:
	- Download de dados pelo `DatasetManager` (pasta `model/dataset.py`)
	- Pré-processamento, normalização e criação de sequências
	- Treinamento do LSTM, logging no MLflow e salvamento dos pesos em `model_files` e do scaler em `scaler_files`

**Notas e observações**
- Os arquivos `last_run_id.txt` e `last_run_date.txt` são usados para rastrear execuções recentes.
- A rota de saúde (`/health`) depende de variáveis globais de modelo/scaler carregadas — ao iniciar a API sem modelos presentes, o endpoint pode indicar que não estão carregados.
- Para adicionar ou atualizar modelos, execute o fluxo de treinamento ou coloque os arquivos em `model_files` com o formato `{TICKER}_{lstm_model_weights.pt2}`.tários para o pré-processamento e inferência.

---
Atualizado em: 2026-05-15
