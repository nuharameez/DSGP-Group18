from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

# Load the trained model
loaded_model = load_model('binary_classification_model_train7.keras')

# Load and preprocess a new image
new_img_path = r"C:\Users\multi\Desktop\KneeKaggle\test\0\9172581L.png"
new_img = image.load_img(new_img_path, target_size=(128, 128))
new_img_array = image.img_to_array(new_img)
new_img_array = cv2.cvtColor(np.array(new_img_array, dtype=np.uint8), cv2.COLOR_BGR2RGB)
new_img_array = cv2.resize(new_img_array, (128, 128))
new_img_array = cv2.GaussianBlur(new_img_array, (5, 5), 0)
new_img_array = np.expand_dims(new_img_array, axis=0)  # Add batch dimension

# Make a prediction
prediction = loaded_model.predict(new_img_array)

# Interpret the prediction
if prediction[0, 0] < 0.5:
    print("The knee is predicted to be normal.")
else:
    print("The knee is predicted to have osteoarthritis.")
