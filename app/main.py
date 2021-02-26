from flask import Flask, request
from os.path import dirname, join
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import urllib.request
import os
import requests
import zipfile
import numpy as np

app= Flask(__name__)
@app.route('/')
def index():
  return "<h1>Welcome to CodingX</h1>"


@app.route("/predict", methods=['POST'])
def predict():
  if request.method == 'POST':
    message = request.form['message']
    data = [message]
  return message

if __name__ == '__main__':
    app.run(debug=False, port=8080, host='0.0.0.0')
