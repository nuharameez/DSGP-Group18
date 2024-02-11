from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
import os
import cv2
import numpy as np


app = Flask(__name__)
CORS(app)

# Load the saved model
loaded_model = load_model(r"C:\Users\chanu\DSGP-Group18\knee_bone_identifier.h5")

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

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image from the request
        image_file = request.files['image']

        # Save the image temporarily
        temp_path = 'temp.png'
        image_file.save(temp_path)

        # Read the QR code from the image
        qr_link = read_qr_code(temp_path)

        if qr_link:
            print(f"QR Code link: {qr_link}")

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

        #return jsonify({"result": result})
            return jsonify({"result": result, "QR_code_link": qr_link})
        else:
            return jsonify({"error": "QR code extraction failed."})

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)