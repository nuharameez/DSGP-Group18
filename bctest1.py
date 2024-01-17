import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Function to load and preprocess a single image
def load_and_preprocess_single_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = cv2.cvtColor(np.array(img_array, dtype=np.uint8), cv2.COLOR_BGR2RGB)
    img_array = cv2.resize(img_array, (224, 224))
    img_array = cv2.GaussianBlur(img_array, (5, 5), 0)
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

# Load the saved VGG16 transfer learning model
saved_model_path = 'vgg16_transfer_learning_model.keras'
loaded_model = load_model(saved_model_path)

# Directory path containing the test images
test_images_dir = r"C:\Users\multi\Desktop\All Folders\knee\test"

# Iterate through each image in the directory
for filename in os.listdir(test_images_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(test_images_dir, filename)

        # Load and preprocess the image
        new_image = load_and_preprocess_single_image(image_path)

        # Make predictions
        prediction = loaded_model.predict(new_image)

        # Convert prediction to class label (assuming binary classification)
        predicted_class = (prediction > 0.5).astype(int)

        # Display results
        class_label = "Normal" if predicted_class[0] == 0 else "Not Normal"
        print(f"Image: {filename}, Prediction Probability: {prediction[0, 0]}, Predicted Class: {class_label}")
