import os
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, send_file
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np
import base64
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

# Load the RandomForestClassifier model for treatment recommendation
rf_model_path = 'random_forest_model_new.pkl'
treatment_model = joblib.load(rf_model_path)

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


# Function to decode QR code and retrieve file path
def decode_qr_code(qr_image):
    qr_detector = cv2.QRCodeDetector()
    data, _, _ = qr_detector.detectAndDecode(qr_image)
    return data.strip() if data else None


def map_severity_to_grading(severity_level):
    if severity_level.startswith('Doubtful'):
        return 1
    elif severity_level.startswith('Minimal'):
        return 2
    elif severity_level.startswith('Moderate'):
        return 3
    elif severity_level.startswith('Extreme'):
        return 4
    else:
        return 0  # Default to Normal


def get_treatments(severity_level, age_category, gender):
    new_data = [[gender, age_category, severity_level]]  # Age, Gender, Severity
    treatment = treatment_model.predict(new_data)
    treatment_labels = {
        0: 'Regular Exercises',
        1: 'Proper lifting techniques',
        2: 'Balanced Diet',
        3: 'Physical Therapy exercises',
        4: 'Aromatherapy'
    }
    treatment = treatment_labels[int(treatment[0])]
    return treatment


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
    age_category = request.form.get('age_category', '')
    gender_str = request.form.get('gender', '')

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if not (age_category and gender_str):
        return jsonify({'error': 'Age category and gender are required'})

    if file and allowed_file(file.filename):
        gender = 0 if gender_str.lower() == 'male' else 1

        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        qr_image = cv2.imread(filename)
        qr_data = decode_qr_code(qr_image)

        if qr_data:
            image_path = qr_data
        else:
            image_path = filename

        img_array = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img_array)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        knee_bone_prediction = knee_bone_model.predict(img_array)
        knee_bone_result = 'Not Bone' if knee_bone_prediction < 0.5 else 'Knee Bone Verified'

        if knee_bone_result == 'Not Bone':
            result = {'knee_bone_result': 'Not a Knee Bone', 'image_path': image_path}
            return jsonify(result)

        knee_prediction = knee_model.predict(img_array)
        knee_result = 'Normal' if knee_prediction < 0.5 else 'Abnormal'

        if knee_result == 'Normal':
            result = {'knee_bone_result': 'Knee Bone Verified', 'normal_result': 'Normal', 'image_path': image_path}
            return jsonify(result)

        severity = get_grade(image_path)
        severity_level = map_severity_to_grading(severity)

        treatments = get_treatments(severity_level, age_category, gender)

        result = {'knee_bone_result': 'Knee Bone Verified', 'normal_result': 'Abnormal', 'severity': severity,
                  'treatments': treatments, 'image_path': image_path}

        with open(image_path, "rb") as img_file:
            img_bytes = img_file.read()
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')

        result['image_base64'] = img_base64

        return jsonify(result)

    else:
        return jsonify({'error': 'File type not allowed'})


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/get-image/<image_name>')
def get_image(image_name):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], image_name))


@app.route('/')
def index():
    return app.send_static_file('index.html')


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
