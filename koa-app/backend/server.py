from flask import Flask, request, jsonify, send_from_directory
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

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

def preprocess_image(file_path):
    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize pixel values
    return img_array

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
        img_array = preprocess_image(filename)

        # Perform knee model inference
        knee_prediction = knee_model.predict(img_array)

        # Map the knee prediction to 'Normal' or 'Abnormal'
        knee_result = 'Normal' if knee_prediction < 0.5 else 'Abnormal'

        # Perform model inference for the knee bone model
        knee_bone_prediction = knee_bone_model.predict(img_array)

        # Map the knee bone prediction to 'Bone' or 'Not Bone'
        knee_bone_result = 'Not Bone' if knee_bone_prediction < 0.5 else 'Bone'

        # Determine normal result based on knee prediction
        normal_result = 'Normal' if knee_result == 'Normal' else 'Not Normal'

        # Print knee and normal results for debugging
        print("Knee Result:", knee_bone_result)  # Updated to print knee bone result
        print("Normal Result:", normal_result)

        return jsonify({'knee_result': knee_result, 'normal_result': normal_result, 'knee_bone_result': knee_bone_result})
    else:
        return jsonify({'error': 'File type not allowed'})




@app.route('/')
def index():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
