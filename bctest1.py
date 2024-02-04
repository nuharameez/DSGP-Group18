import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Assuming 'knee_model.h5' is in the current working directory
model_path = 'knee_model1.h5'

# Load the saved model
loaded_model = load_model(model_path)

# Replace this with the path to your test image folder
test_folder_path = r"C:\Users\multi\Desktop\All Folders\knee\test"

# Mapping for printing
class_mapping = {0: "normal", 1: "osteo"}

# Iterate through each image in the folder
for image_filename in os.listdir(test_folder_path):
    # Construct the full path to the image
    image_path = os.path.join(test_folder_path, image_filename)

    # Load and preprocess the test image
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Rescale pixel values

    # Predict using the loaded model
    prediction = loaded_model.predict(img_array)
    prediction_class = (prediction > 0.5).astype(int)

    # Convert the prediction_class array to a scalar (integer)
    prediction_class_scalar = prediction_class.item()

    # Map predictions to class names
    prediction_class_name = class_mapping[prediction_class_scalar]

    # Print the result
    print(f"Image: {image_filename}, Prediction: {prediction_class_name}")
