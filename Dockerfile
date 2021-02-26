FROM avoli/tensorflow-cpu
RUN pip install -r requirements.txt

EXPOSE 8080

CMD python3 /app/app.py
