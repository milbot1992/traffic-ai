from flask import Flask, request, jsonify
import cv2
import numpy as np
import os
import tensorflow as tf

app = Flask(__name__)

# TensorFlow model loading
model = tf.keras.models.load_model('model.h5.keras')

# Constants
IMG_WIDTH = 30
IMG_HEIGHT = 30

# Define route for image classification
@app.route('/classify', methods=['POST'])

def classify_image():
    """
    Classifies an image uploaded via an HTTP request.

    This function handles a file upload, preprocesses the image,
    and uses a pre-trained model to predict the class of the image.
    It returns a JSON response containing the predicted class.

    Returns:
        A Flask JSON response object with either an error message if no file is uploaded
        or the predicted class of the image.
    """
    # Check if the request contains a file
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    # Get the uploaded file
    file = request.files['file']

    # Read and preprocess the image
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = np.expand_dims(img, axis=0)

    # Perform classification using the model
    predictions = model.predict(img)

    # Get the predicted class label
    predicted_class = np.argmax(predictions)

    return jsonify({'predicted_class': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
