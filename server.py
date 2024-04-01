"""import os
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np

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

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/')
def index():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)"""
from sklearn.preprocessing import LabelEncoder

"""import os
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import pandas as pd

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

# Load treatment dataset
treatment_dataset_path = 'Treatement dataset1.csv'
treatment_df = pd.read_csv(treatment_dataset_path)

# Function to recommend treatments for a specific grade
def recommend_treatments_for_grade(grade):
    # Extract the grade number from the input grade string
    grade_number = int(grade.split()[-1])
    # Find treatments for the extracted grade number
    grade_treatments = treatment_df[treatment_df['Level'].str.contains(f'Grade {grade_number}')]
    if not grade_treatments.empty:
        return grade_treatments['Treatment Recommendation'].tolist()
    else:
        return []


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

        # Get treatment recommendations based on severity
        treatments = recommend_treatments_for_grade(severity)

        result = {'knee_bone_result': 'Knee Bone Verified', 'normal_result': 'Abnormal', 'severity': severity, 'treatments': treatments, 'image_path': image_url}
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
    app.run(debug=True)"""

import os
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, url_for, render_template
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

# Load the RandomForestClassifier model for treatment recommendation
rf_model_path = 'random_forest_model_new.pkl'
treatment_model = joblib.load(rf_model_path)

# Define the categories
categories = ['Doubtful: KL grading- 1', 'Mild: KL grading- 2', 'Moderate: KL grading- 3', 'Severe: KL grading- 4']

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

# Function to map severity levels to grading levels used in the dataset
def map_severity_to_grading(severity_level):
    if severity_level.startswith('Normal'):
        return 0
    elif severity_level.startswith('Doubtful'):
        return 1
    elif severity_level.startswith('Mild'):
        return 2
    elif severity_level.startswith('Moderate'):
        return 3
    elif severity_level.startswith('Severe'):
        return 4
    else:
        return None  # Handle unknown severity levels

def get_treatments(treatment_model, severity_level, age_category, gender):
    # For example, let's create a single new data point with random values
    new_data = [[gender, age_category, severity_level]]  # Age, Gender, Severity

    # Use the loaded model to make predictions on the new data
    treatment = treatment_model.predict(new_data)
    treatment_labels = {
        0: 'Regular Exercises',
        1: 'Proper lifting techniques',
        2: 'Balanced Diet',
        3: 'Physical Therapy exercises',
        4: 'Aromatherapy'
    }
    treatment = treatment_labels[int(treatment[0])]
    print("Filtered Treatments:", treatment)
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

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        # Save the file
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        print("File saved at:", filename)

        # Load and preprocess the image for the knee model
        img_array = image.load_img(filename, target_size=(224, 224))
        img_array = image.img_to_array(img_array)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize pixel values

        # Perform knee bone identification
        knee_bone_prediction = knee_bone_model.predict(img_array)

        # Map the knee bone prediction to 'Bone' or 'Not Bone'
        knee_bone_result = 'Not Bone' if knee_bone_prediction < 0.5 else 'Knee Bone Verified'

        print("Knee Bone Result:", knee_bone_result)

        if knee_bone_result == 'Not Bone':
            # If not a knee bone, return the result
            result = {'knee_bone_result': 'Not a Knee Bone'}
            print(result)
            return jsonify(result)

        # Perform knee model inference
        knee_prediction = knee_model.predict(img_array)

        # Map the knee prediction to 'Normal' or 'Abnormal'
        knee_result = 'Normal' if knee_prediction < 0.5 else 'Abnormal'

        print("Knee Result:", knee_result)

        if knee_result == 'Normal':
            # If the knee bone is normal, return the result
            result = {'knee_bone_result': 'Knee Bone Verified', 'normal_result': 'Normal'}
            print(result)
            return jsonify(result)

        # If knee bone is abnormal, determine severity
        severity = get_grade(filename)

        print("Severity:", severity)

        # Get age and gender from user input
        age_category = request.form.get('age_category')
        gender = request.form.get('gender')

        # Map severity to corresponding numbers
        severity_level = map_severity_to_grading(severity)

        if severity_level is None:
            return jsonify({'error': 'Unknown severity level'})

        # Get treatment recommendations based on severity, age, and gender
        treatments = get_treatments(treatment_model, severity_level, age_category, gender)

        result = {'knee_bone_result': 'Knee Bone Verified', 'normal_result': 'Abnormal',
                  'severity': severity, 'treatments': treatments}

        print("Result:", result)
        return jsonify(result)

    else:
        return jsonify({'error': 'File type not allowed'})


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/recommend', methods=['PUT'])
def recommend_treatments():
    # Get age category and gender from user input
    age_category = request.form.get('age_category')
    gender = request.form.get('gender')

    if not age_category or not gender:
        return jsonify({'error': 'Age category or gender not provided'})

    # Get severity level from the results
    result = request.json
    severity = result.get('severity')

    severity_types = {
        'Normal: KL grading- 0': 0,
        'Doubtful: KL grading- 1': 1,
        'Mild: KL grading- 2': 2,
        'Moderate: KL grading- 3': 3,
        'Severe: KL grading- 4': 4
    }

    # Map severity to corresponding numbers
    severity_level = severity_types.get(severity)

    if severity_level is None:
        return jsonify({'error': 'Severity level not found in results'})

    # Get treatments based on severity, age, and gender
    treatments = get_treatments(treatment_model, severity_level, age_category, gender)

    print("Recommended Treatments:", treatments)  # Print treatments to console

    return jsonify({'treatments': treatments})


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
