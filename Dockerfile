FROM python:3.12

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y git && apt-get clean

RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Make sure Python can import from src/
ENV PYTHONPATH="/app"

ENTRYPOINT ["python", "cli.py"]
