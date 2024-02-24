import os
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np
from flask_cors import CORS
import requests

app = Flask(__name__, static_folder="build", static_url_path="/")
CORS(app)

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


def read_qr_code(image_path):
    # Read the image
    img = cv2.imread(image_path)

    # Initialize the QR code detector
    qr_code_detector = cv2.QRCodeDetector()

    # Detect and decode the QR code
    retval, decoded_info, points, straight_qrcode = qr_code_detector.detectAndDecodeMulti(img)

    if retval:
        # Extract the link
        link = decoded_info[0]
        print(f"QR Code link: {link}")
        return link
    else:
        print("No QR code found in the image.")
        return None

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response
"""""
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

        # Load and preprocess the image for the knee model
        img_array = image.load_img(filename, target_size=(224, 224))
        img_array = image.img_to_array(img_array)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize pixel values

        # Get the URL of the uploaded image
        image_url = url_for('uploaded_file', filename=file.filename)

        # Perform knee bone identification
        knee_bone_prediction = knee_bone_model.predict(img_array)

        # Map the knee bone prediction to 'Bone' or 'Not Bone'
        knee_bone_result = 'Not Bone' if knee_bone_prediction < 0.5 else 'Knee Bone Verified'

        if knee_bone_result == 'Not Bone':
            # If not a knee bone, return the result
            result = {'knee_bone_result': 'Not a Knee Bone', 'image_path': image_url}
            print(result)
            return jsonify(result)

        # Perform knee model inference
        knee_prediction = knee_model.predict(img_array)

        # Map the knee prediction to 'Normal' or 'Abnormal'
        knee_result = 'Normal' if knee_prediction < 0.5 else 'Abnormal'

        if knee_result == 'Normal':
            # If the knee bone is normal, return the result
            result = {'knee_bone_result': 'Knee Bone Verified', 'normal_result': 'Normal', 'image_path': image_url}
            print(result)
            return jsonify(result)

        # If knee bone is abnormal, determine severity
        severity = get_grade(filename)

        result = {'knee_bone_result': 'Knee Bone Verified', 'normal_result': 'Abnormal', 'severity': severity, 'image_path': image_url}
        print(result)
        return jsonify(result)

    else:
        return jsonify({'error': 'File type not allowed'})
"""""

@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'qr_file' in request.files:
        qr_file = request.files['qr_file']
        if qr_file and allowed_file(qr_file.filename):
            # Save the QR code file
            qr_filename = os.path.join(app.config['UPLOAD_FOLDER'], qr_file.filename)
            qr_file.save(qr_filename)

            # Decode QR code
            qr_link = read_qr_code(qr_filename)
            if qr_link:
                return jsonify({"QR_code_link": qr_link})
            else:
                return jsonify({'error': 'Failed to decode QR code'})

    if 'xray_file' in request.files:
        xray_file = request.files['xray_file']
        if xray_file and allowed_file(xray_file.filename):
            # Save the X-ray file
            xray_filename = os.path.join(app.config['UPLOAD_FOLDER'], xray_file.filename)
            xray_file.save(xray_filename)

            # Load and preprocess the image for the knee model
            img_array = image.load_img(xray_filename, target_size=(224, 224))
            img_array = image.img_to_array(img_array)
            img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize pixel values

            # Perform knee bone identification
            knee_bone_prediction = knee_bone_model.predict(img_array)

            # Map the knee bone prediction to 'Bone' or 'Not Bone'
            knee_bone_result = 'Not Bone' if knee_bone_prediction < 0.5 else 'Knee Bone Verified'

            if knee_bone_result == 'Not Bone':
                # If not a knee bone, return the result
                result = {'knee_bone_result': 'Not a Knee Bone', 'image_path': url_for('uploaded_file', filename=xray_file.filename)}
                print(result)
                return jsonify(result)

            # Perform knee model inference
            knee_prediction = knee_model.predict(img_array)

            # Map the knee prediction to 'Normal' or 'Abnormal'
            knee_result = 'Normal' if knee_prediction < 0.5 else 'Abnormal'

            if knee_result == 'Normal':
                # If the knee bone is normal, return the result
                result = {'knee_bone_result': 'Knee Bone Verified', 'normal_result': 'Normal', 'image_path': url_for('uploaded_file', filename=xray_file.filename)}
                print(result)
                return jsonify(result)

            # If knee bone is abnormal, determine severity
            severity = get_grade(xray_filename)

            result = {'knee_bone_result': 'Knee Bone Verified', 'normal_result': 'Abnormal', 'severity': severity, 'image_path': url_for('uploaded_file', filename=xray_file.filename)}
            print(result)
            return jsonify(result)

    return jsonify({'error': 'No file part or invalid file'})


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/')
def index():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
