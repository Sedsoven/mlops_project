FROM python:3.10

WORKDIR /app
COPY vertex_train.py .

RUN pip install scikit-learn

CMD ["python", "vertex_train.py"]
