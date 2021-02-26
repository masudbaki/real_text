FROM python:3.7.3-stretch
RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install -r requirements.txt
EXPOSE 8080

CMD python3 /app/app.py
