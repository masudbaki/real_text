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
  print(os.getcwd())
  print(pickle.format_version)
  #tokenizer_url = "https://drive.google.com/uc?export=download&id=1-5PXkN3D8uXTvtd6rL-lDI1pO3CvWdfv"
  #urllib.request.urlretrieve(tokenizer_url,"tokenizer.pickle")
  #myfile = requests.get("https://drive.google.com/uc?export=download&id=1-5PXkN3D8uXTvtd6rL-lDI1pO3CvWdfv")
  ##open('tokenizer.pickle', 'wb').write(myfile.content)
  f_result_string = []
  filename = "/app/tokenizer.pickle"
  
  with zipfile.ZipFile("tokenizer.zip", 'r') as zip_ref:
    zip_ref.extractall("/app")
  print(os.listdir())
  with open(filename, 'rb') as handle:
    tokenizer = pickle.load(handle)
  texts = tokenizer.texts_to_sequences(data)
  processed_string = pad_sequences(texts, maxlen=859, padding='post')
    # Load the TFLite model and allocate tensors.
  model_url = "https://drive.google.com/uc?export=download&id=12W9mcMYKEDQIzTaDlzpwRym66rSPtAg5"
  urllib.request.urlretrieve(model_url,"FT_model.tflite")
  interpreter = tf.lite.Interpreter(model_path="FT_model.tflite")
  interpreter.allocate_tensors()
  print("model Loaded")
    # Get input and output tensors.
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
    # Test the model on random input data.
  input_shape = input_details[0]['shape']
  input_data = np.array(processed_string, dtype=np.float32)
  interpreter.set_tensor(input_details[0]['index'], input_data)
  interpreter.invoke()
  output_data = interpreter.get_tensor(output_details[0]['index'])
  print(output_data[0][0])
  if round(output_data[0][0]) == 0:
    f_result_string.append("F")
  else:
    f_result_string.append("T")
  print(f_result_string)
  
 
  
  #3rd model
  
  
  model_url = "https://drive.google.com/uc?export=download&id=1-BZUnV6zavnvDfVEGhKHl9PeUdqFykLY"
  urllib.request.urlretrieve(model_url,"JP_model.tflite")
  interpreter = tf.lite.Interpreter(model_path="JP_model.tflite")
  interpreter.allocate_tensors()
  print("model Loaded")
    # Get input and output tensors.
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
    # Test the model on random input data.
  input_shape = input_details[0]['shape']
  input_data = np.array(processed_string, dtype=np.float32)
  interpreter.set_tensor(input_details[0]['index'], input_data)
  interpreter.invoke()
  output_data = interpreter.get_tensor(output_details[0]['index'])
  print(output_data[0][0])
  if round(output_data[0][0]) == 0:
    f_result_string.append("J")
  else:
    f_result_string.append("P")
  print(f_result_string)
  
  #4th model
  
  model_url = "https://drive.google.com/uc?export=download&id=1-Bdta2WiKax-3RZ0EqHAx42RD7JfBq40"
  urllib.request.urlretrieve(model_url,"NS_model.tflite")
  interpreter = tf.lite.Interpreter(model_path="NS_model.tflite")
  interpreter.allocate_tensors()
  print("model Loaded")
    # Get input and output tensors.
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
    # Test the model on random input data.
  input_shape = input_details[0]['shape']
  input_data = np.array(processed_string, dtype=np.float32)
  interpreter.set_tensor(input_details[0]['index'], input_data)
  interpreter.invoke()
  output_data = interpreter.get_tensor(output_details[0]['index'])
  print(output_data[0][0])
  if round(output_data[0][0]) == 0:
    f_result_string.append("N")
  else:
    f_result_string.append("S")
  print(f_result_string)
  
  return f'{f_result_string[0]} {f_result_string[1]} {f_result_string[2]} {f_result_string[3]}'


