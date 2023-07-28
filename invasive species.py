import tensorflow as tf
import requests
from PIL import Image
from io import BytesIO
import numpy as np
from PyQt5.uic import *
from PyQt5.QtWidgets import *

# Load the exported TensorFlow model
model_path = "keras_model.h5"  # Update this with the path to your model
model = tf.keras.models.load_model(model_path)

# Function for image classification
def predict_local_image(model, image_path):
    img = Image.open(image_path)

    # Preprocess the image
    img = img.resize((224, 224))  # Adjust the size according to the model's input shape
    x = np.array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    # Make the prediction
    prediction = model.predict(x)
    class_index = np.argmax(prediction)
    classes = ['Invasive Animal', 'Non-Invasive Animal']  # Replace with your own class labels
    predicted_class = classes[class_index]
    confidence = prediction[0][class_index]

    return predicted_class, confidence

def main():
    image_path = win.url.text()
    label, confidence = predict_local_image(model, image_path)
    win.result.setText(f"Prediction: {label}, Confidence: {confidence:.2f}")


app = QApplication([])
win = loadUi("C:/Users/rbouh/Desktop/detector/detector.ui")
win.show()
win.btn.clicked.connect(main)
app.exec_()