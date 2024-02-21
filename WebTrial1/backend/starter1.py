""""
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
"""""

# Load the saved model
#loaded_model = load_model(r"C:\Users\chanu\DSGP-Group18\knee_bone_identifier.h5")
""""
import cv2
import os
import requests

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

def download_image_from_url(url, save_directory):
    # Create the save directory if it doesn't exist
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    try:
        # Send a GET request to the URL
        response = requests.get(url)
        if response.status_code == 200:
            # Extract the filename from the URL
            filename = url.split("/")[-1]
            # Save the image to the specified directory
            with open(os.path.join(save_directory, filename), 'wb') as f:
                f.write(response.content)
            print(f"Image downloaded successfully: {os.path.join(save_directory, filename)}")
            return os.path.join(save_directory, filename)
        else:
            print("Failed to download image. Status code:", response.status_code)
            return None
    except Exception as e:
        print("An error occurred while downloading the image:", str(e))
        return None

# Example usage:
#image_path = "WebTrial1/backend/"  # Replace this with the path to your image
#link = read_qr_code(image_path)
save_directory = "WebTrial1/backend"  # Choose your folder name here
#if link:
 #   download_image_from_url(link, save_directory)





from flask import Flask, request, jsonify
#from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Load the downloaded image
        image_path = "backend/downloaded_image.jpg"  # Update this path with the path where you saved the downloaded image
        if not os.path.exists(image_path):
            return jsonify({"error": "Downloaded image not found."}), 404

        # Load and preprocess the image
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Rescale to match the training data

        # Make predictions using the loaded model
        predictions = loaded_model.predict(img_array)

        # Display the result
        result = "Knee Bone" if predictions[0][0] > 0.5 else "Not a Knee Bone"

        print("Prediction result:", result)

        return jsonify({"result": result})

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)

"""""
import os
from flask import Flask, request, jsonify
import requests
from PIL import Image
from io import BytesIO
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
#from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

app = Flask(__name__)

# Load pre-trained model
model =  load_model(r"C:\Users\chanu\DSGP-Group18\knee_bone_identifier.h5")

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
    # Check if the POST request has the file part
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['image']

    # Check if the file is empty
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Check if the file is a QR code
    img = Image.open(file)
    image_path = "temp_image.png"
    #img = img.convert("RGB")  # Convert to RGB mode
    img.save(image_path)
    url = read_qr_code(image_path)

    if url is None:
        return jsonify({'error': 'Uploaded image is not a QR code'})

    # Download image from URL
    response = requests.get(url)
    #image = Image.open(BytesIO(response.content))
    # Debugging statements
    print("Response status code:", response.status_code)
    print("Response content type:", response.headers.get('content-type'))

    if response.status_code == 200 and response.headers.get('content-type', '').startswith('image'):

        downloaded_image_path = "temp_image_downloaded.jpg"
        with open(downloaded_image_path, 'wb') as f:
            f.write(response.content)

        # Check if the file was saved correctly
        if os.path.exists(downloaded_image_path):

            # Load the downloaded image
            downloaded_image = Image.open(downloaded_image_path)

            # Preprocess image for prediction
            downloaded_image = downloaded_image.resize((224, 224))  # Adjust size according to your model's input shape
            downloaded_image = np.expand_dims(downloaded_image, axis=0)
            downloaded_image = downloaded_image / 255.0  # Normalize if needed

        #image = preprocess_input(image)

            # Make prediction
            predictions = model.predict(downloaded_image)
            result = "Knee Bone" if predictions[0][0] > 0.5 else "Not a Knee Bone"
            #decoded_predictions = decode_predictions(predictions, top=1)[0]

        # Return result
            return jsonify({'result': result})
        else:
            return jsonify({'error': 'Failed to download or invalid image from the URL'})
    else:
        return jsonify({'error': 'Failed to download or invalid image from the URL'})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
