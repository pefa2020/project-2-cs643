FROM python:3.10-slim

# Install Java (required by Spark)
RUN apt-get update && \
    apt-get install -y openjdk-21-jre && \
    apt-get clean

# Spark temp directory
RUN mkdir -p /tmp/spark && chmod -R 777 /tmp/spark

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# ENTRYPOINT so TA's extra argument becomes sys.argv[1]
ENTRYPOINT ["python3", "predict.py"]
