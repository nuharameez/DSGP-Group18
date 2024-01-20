from joblib import load
import cv2
import numpy as np

# Load the saved decision tree model
model_filename = "decision_tree_model.joblib"
loaded_model = load(model_filename)

# Load and preprocess the input image
input_image_path = r"C:\Users\MSI\Downloads\IIT STUFF\CM 2603 DS\CW implementation testing\DATASETS\train\2\9755634L.png"
input_img = cv2.imread(input_image_path)
input_img = cv2.resize(input_img, (224, 224))
input_img = input_img / 255.0

input_img_flattened = input_img.reshape(1, -1)

# Make predictions using the loaded model
prediction = loaded_model.predict(input_img_flattened)

classification_grade = f"KL{prediction[0]}"
print(f"Classification Grade: {classification_grade}")