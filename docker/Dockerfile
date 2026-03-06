FROM python:3.11-slim

WORKDIR /recommender_app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/recommender_app

CMD ["python","main/main_train.py"]