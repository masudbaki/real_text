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
import requests
from os import path

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
  
  if path.exists("models.zip") == False:
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params = { 'id' : '1ZgyHHApsrjBG346uJ-CJ3ucNnNpf_tK8' }, stream = True)
    for key, value in response.cookies.items():
      if key.startswith('download_warning'):
        token = value
    if token:
      params = { 'id' : '1ZgyHHApsrjBG346uJ-CJ3ucNnNpf_tK8', 'confirm' : token }
      response = session.get(URL, params = params, stream = True)
    CHUNK_SIZE = 32768
    with open("models.zip", "wb") as f:
      for chunk in response.iter_content(CHUNK_SIZE):
        if chunk: # filter out keep-alive new chunks
          f.write(chunk)

  f_result_string = []
  print(os.getcwd())
  with zipfile.ZipFile("models.zip", 'r') as zip_ref:
    zip_ref.extractall("/app")
  print("zipping done")
  os.remove("models.zip")
  print("models.zip removed")
  
  print(os.listdir())
  with open("/app/model/tokenizer.pickle", 'rb') as handle:
    tokenizer = pickle.load(handle)
  texts = tokenizer.texts_to_sequences(data)
  processed_string = pad_sequences(texts, maxlen=859, padding='post')
    # Load the TFLite model and allocate tensors.
    
    #1st model
    
  interpreter = tf.lite.Interpreter(model_path="/app/model/FT_model.tflite")
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
  
  #2nd model
  
  interpreter = tf.lite.Interpreter(model_path="/app/model/IE_model.tflite")
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
    f_result_string.append("I")
  else:
    f_result_string.append("E")
  print(f_result_string)
  
 
  
  #3rd model
  
 
  interpreter = tf.lite.Interpreter(model_path="/app/model/JP_model.tflite")
  interpreter.allocate_tensors()
  print("jp model Loaded")
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
  
 
  interpreter = tf.lite.Interpreter(model_path="/app/model/NS_model.tflite")
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
