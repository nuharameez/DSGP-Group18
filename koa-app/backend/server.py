# server.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS  # Import CORS
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__, static_folder="build", static_url_path="/")
CORS(app)  # Enable CORS

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your pre-trained model here
# Example: InceptionV3 model
model_path = r"C:\Users\multi\demo1\backend\knee_model1.h5"
loaded_model = load_model(model_path)

def preprocess_image(file_path):
    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize pixel values
    return img_array

@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        # Save the file
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        # Load and preprocess the image for the model
        img_array = preprocess_image(filename)

        # Perform model inference
        prediction = loaded_model.predict(img_array)

        # Map the prediction to 'Normal' or 'Abnormal'
        prediction_class = (prediction > 0.5).astype(int).item()
        result = 'Normal' if prediction_class == 0 else 'Abnormal'

        return jsonify({'result': result})
    else:
        return jsonify({'error': 'File type not allowed'})

@app.route('/')
def index():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
