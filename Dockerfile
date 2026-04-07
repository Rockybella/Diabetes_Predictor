FROM python:3.12-slim
WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

# Hugging Face Spaces always use port 7860
EXPOSE 7860

# Gunicorn command for FastAPI
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "app:app", "--bind", "0.0.0.0:7860"]
