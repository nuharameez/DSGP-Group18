import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the trained model
model = load_model('knee_xray_classifier.h5')

# Preprocess a new X-ray image
img_path = r"C:\Users\multi\Desktop\knee\validate\normal\9075745R.png"
img_width, img_height = 150, 150  # Replace with the dimensions used during training

img = load_img(img_path, target_size=(img_width, img_height))  # Use the same dimensions as training
img_array = img_to_array(img)
img_batch = np.expand_dims(img_array, axis=0)

# Make a prediction
prediction = model.predict(img_batch)

# Interpret the prediction
if prediction[0][0] > 0.5:
    print("The X-ray is likely normal.")
else:
    print("The X-ray likely shows osteoarthritis.")
