FROM python:3.12.3

WORKDIR /tech_challenge_4_IA

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .

WORKDIR /tech_challenge_4_IA/app

# RUN flask db init
# RUN flask db migrate
# RUN flask db upgrade

CMD ["python3", "app.py"]