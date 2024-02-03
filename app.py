from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
import joblib
import os
from werkzeug.utils import secure_filename
from keras.applications.vgg16 import VGG16, preprocess_input

# Load pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


app = Flask(__name__)

# Load the pre-trained model
model = load_model('VGG16KFold3.h5')

# Load the label encoder
label_encoder = joblib.load('label_encoder_VGG16KFold3.joblib')

# Function to preprocess the uploaded image
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    # Extract features using VGG16
    features = base_model.predict(img)
    features_flat = features.reshape(1, -1)

    return features_flat


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        file.save(file_path)

        # Preprocess the uploaded image
        img = preprocess_image(file_path)

        # Make predictions
        predictions = model.predict(img)
        predicted_class = label_encoder.classes_[np.argmax(predictions)]

        # Delete the uploaded file after processing
        os.remove(file_path)

        return jsonify({'class': predicted_class})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Create the 'uploads' directory if it doesn't exist
    os.makedirs('uploads', exist_ok=True)

    app.run(debug=True)
