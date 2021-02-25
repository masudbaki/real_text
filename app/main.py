from flask import Flask, request
from os.path import dirname, join
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import urllib.request
import os
import requests

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
  #tokenizer_url = "https://drive.google.com/uc?export=download&id=1-5PXkN3D8uXTvtd6rL-lDI1pO3CvWdfv"
  #urllib.request.urlretrieve(tokenizer_url,"tokenizer.pickle")
  #myfile = requests.get("https://drive.google.com/uc?export=download&id=1-5PXkN3D8uXTvtd6rL-lDI1pO3CvWdfv")
  #open('tokenizer.pickle', 'wb').write(myfile.content)
  filename = "tokenizer.pickle"
  with open(filename, 'rb') as handle:
    tokenizer = pickle.load(handle)
  texts = tokenizer.texts_to_sequences(data)
  processed_string = pad_sequences(texts, maxlen=859, padding='post')
    # Load the TFLite model and allocate tensors.
  model_url = "https://drive.google.com/uc?export=download&id=12W9mcMYKEDQIzTaDlzpwRym66rSPtAg5"
  urllib.request.urlretrieve(model_url,"FT_model.tflite")
  interpreter = tf.lite.Interpreter(model_path="FT_model.tflite")
  interpreter.allocate_tensors()
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
  return message
    
  
  return message

