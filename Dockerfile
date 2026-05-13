FROM python:3.12.3

WORKDIR /tech_challenge_4_IA

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .

ENV PYTHONPATH=/tech_challenge_4_IA

CMD ["python", "-m", "api.app"]