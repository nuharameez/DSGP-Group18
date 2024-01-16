import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import os

# Load the saved model
model_path = 'binary_classification_model_updated.keras'
try:
    model = load_model(model_path)
except Exception as e:
    print(f"Error loading the model: {e}")

# Function to preprocess a single image
def preprocess_image(img_path):
    img = load_img(img_path, target_size=(128, 128))
    img_array = img_to_array(img)
    img_array = cv2.cvtColor(np.array(img_array, dtype=np.uint8), cv2.COLOR_BGR2RGB)
    img_array = cv2.resize(img_array, (128, 128))
    img_array = cv2.GaussianBlur(img_array, (5, 5), 0)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Test folder path
test_folder_path = r"C:\Users\multi\Desktop\knee\test"

# Loop through images in the folder
for filename in os.listdir(test_folder_path):
    img_path = os.path.join(test_folder_path, filename)

    # Preprocess the image
    preprocessed_image = preprocess_image(img_path)

    try:
        # Make a prediction
        prediction = model.predict(preprocessed_image)

        # Interpret the prediction
        class_label = "Osteoarthritis" if prediction > 0.5 else "Normal"
        confidence = prediction[0, 0]

        print(f"Image: {filename}, Predicted class: {class_label}, Confidence: {confidence}")

    except Exception as e:
        print(f"Error predicting image {filename}: {e}")
