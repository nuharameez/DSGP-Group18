import os
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import pandas as pd
import joblib

app = Flask(__name__, static_folder="build", static_url_path="/")

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load knee model
knee_model_path = 'knee_model1.h5'
knee_model = load_model(knee_model_path)

# Load knee bone identifier model
knee_bone_model_path = 'knee_bone_identifier.h5'
knee_bone_model = load_model(knee_bone_model_path)

# Load the saved model for severity determination
severity_model_path = 'Custom_CNN_with_VGG16.h5'
severity_model = load_model(severity_model_path)

# Define the categories
categories = ['Doubtful: KL grading- 1', 'Minimal: KL grading- 2', 'Moderate: KL grading- 3', 'Extreme: KL grading- 4']

# Load treatments dataframe
df = pd.read_csv('treatments.csv')

# Load the trained model
severity_prediction_model = joblib.load('rf1.pkl')

# Function to preprocess uploaded image
def preprocess_image(image_path):
    img_size = 256
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (img_size, img_size))
    return np.expand_dims(resized / 255.0, axis=0)

# Function to get the grade for the image
def get_grade(image_path):
    preprocessed_img = preprocess_image(image_path)
    prediction = severity_model.predict(preprocessed_img)
    category_index = np.argmax(prediction)
    return categories[category_index]

# Function to decode QR code and retrieve file path
def decode_qr_code(qr_image):
    qr_detector = cv2.QRCodeDetector()
    data, _, _ = qr_detector.detectAndDecode(qr_image)
    return data.strip() if data else None

# Function to get treatments for a given severity level
def get_treatments(severity_level):
    treatments = df[df['Severity'] == severity_level]['Treatment']
    return treatments.tolist()

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

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

        # Decode QR code
        qr_image = cv2.imread(filename)
        qr_data = decode_qr_code(qr_image)

        print("QR Code Data:", qr_data)  # Print QR code data

        if qr_data:
            image_path = qr_data
        else:
            image_path = filename

        print("Image Path:", image_path)  # Print image path

        # Load and preprocess the image for the knee model
        img_array = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img_array)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize pixel values

        # Perform knee bone identification
        knee_bone_prediction = knee_bone_model.predict(img_array)

        # Map the knee bone prediction to 'Bone' or 'Not Bone'
        knee_bone_result = 'Not Bone' if knee_bone_prediction < 0.5 else 'Knee Bone Verified'

        if knee_bone_result == 'Not Bone':
            # If not a knee bone, return the result
            result = {'knee_bone_result': 'Not a Knee Bone', 'image_path': image_path}
            print(result)
            return jsonify(result)

        # Perform knee model inference
        knee_prediction = knee_model.predict(img_array)

        # Map the knee prediction to 'Normal' or 'Abnormal'
        knee_result = 'Normal' if knee_prediction < 0.5 else 'Abnormal'

        if knee_result == 'Normal':
            # If the knee bone is normal, determine severity
            severity = 0
        else:
            # If knee bone is abnormal, determine severity
            severity_level = get_grade(image_path)
            severity = int(severity_level.split("-")[-1].strip())

        # Get treatments for the determined severity level
        treatments = get_treatments(severity)

        result = {'knee_bone_result': 'Knee Bone Verified', 'normal_result': knee_result, 'severity': severity,
                  'treatments': treatments, 'image_path': image_path}
        print(result)
        return jsonify(result)

    else:
        return jsonify({'error': 'File type not allowed'})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/')
def index():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
