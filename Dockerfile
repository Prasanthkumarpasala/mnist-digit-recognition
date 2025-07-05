FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY digit_recognition_api.py .
EXPOSE 5000
CMD ["python", "digit_recognition_api.py"]