from flask import Flask, request, jsonify
from flask_cors import CORS
from flask import send_file

import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
import os
import cv2
import numpy as np
import requests

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

"""""
def download_image(url, save_path):
    try:
        # Send a GET request to the URL
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Write the image content to a file
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
        else:
            print("Failed to download image.")
            return False
    except Exception as e:
        print("Error downloading image:", str(e))
        return False
"""""

"""""
def download_image(url, save_folder, filename):
    try:
        # Ensure the save folder exists, create it if not
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        # Construct the full save path
        save_path = os.path.join(save_folder, filename)

        # Send a GET request to the URL
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Write the image content to a file
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True, save_path  # Return success and the saved file path
        else:
            print("Failed to download image.")
            return False, None
    except Exception as e:
        print("Error downloading image:", str(e))
        return False, None

save_folder = 'WebTrial/backend'
filename = 'downloaded_image.png'
"""""
"""""
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

            # Download the image from the QR code link
            image_path = 'downloaded_image.jpg'
            if download_image(qr_link, image_path):

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
                #os.remove(temp_path)

        # Remove the downloaded image file
                #os.remove(image_path)

                print("Prediction result:", result)

        #return jsonify({"result": result})
                return jsonify({"result": result, "QR_code_link": qr_link})
                #return jsonify({"QR_code_link": qr_link})
            else:
                return jsonify({"error": "Failed to download image from QR code link."})

        else:
            return jsonify({"error": "QR code extraction failed."})

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)})
"""""

@app.route('/decode_qr_code', methods=['POST'])
def decode_qr_code():
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

            return jsonify({"QR_code_link": qr_link})
        else:
            return jsonify({"error": "QR code extraction failed."})

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)})

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

@app.route('/image')
def get_image():
    # Send the image file in the response
    return send_file('temp.png', mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
