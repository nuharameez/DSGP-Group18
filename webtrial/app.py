"""""


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Call your ML model function here
        #prediction = run_ml_model("models/knee_bone_identifier.h5")
        prediction = model.predict(filepath)

        return jsonify({'prediction': prediction})

    return jsonify({'error': 'Invalid file format'})

def run_ml_model(image_path):
    # Implement code to load and run your ML model
    # Replace the following line with actual model inference code
    # For demonstration purposes, let's assume the model always predicts 'Knee Bone'
    return 'Knee Bone'

if __name__ == '__main__':
    app.run(debug=True)
"""""
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
import os
#import cv2
import numpy as np

app = Flask(__name__)
CORS(app)

# Load the saved model
loaded_model = load_model(r"C:\Users\chanu\DSGP-Group18\knee_bone_identifier.h5")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image from the request
        image_file = request.files['image']

        # Save the image temporarily
        temp_path = 'temp.png'
        image_file.save(temp_path)

        # Load and preprocess the image
        img = image.load_img(temp_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Rescale to match the training data

        # Make predictions using the loaded model
        predictions = loaded_model.predict(img_array)

        # Display the result
        result = "Knee Bone" if predictions[0][0] > 0.5 else "Not a Knee Bone"

        # Remove the temporary image file
        os.remove(temp_path)

        print("Prediction result:", result)

        return jsonify({"result": result})

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)

"""""
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image from the request
        image = request.files['image']

        # Decode the image using cv2
        img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_UNCHANGED)

        # Preprocess the image (adjust as needed based on your training preprocessing)
        img = cv2.resize(img, (224, 224))
        img = np.expand_dims(img, axis=0) / 255.0

        # Make a prediction
        prediction = model.predict(img)

        # Assuming binary classification (knee bone or not)
        result = "Knee Bone" if prediction[0][0] > 0.5 else "Not a Knee Bone"

        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"error": str(e)})
"""""