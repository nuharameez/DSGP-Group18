import cv2
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import load_model
from keras.preprocessing import image

# Load pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Load the trained model
model = load_model(r'C:\Users\MSI\Downloads\IIT STUFF\CM 2603 DS\DSGP-Group18\VGG16KFold.h5')


# Function to preprocess a single image and make predictions
def predict_grade(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = preprocess_input(img)

    # Extract features using VGG16
    features = base_model.predict(img)
    features_flat = features.reshape(features.shape[0], -1)

    # Make predictions using the trained model
    predicted_class = np.argmax(model.predict(features_flat), axis=1)[0]

    return predicted_class + 1  # Adding 1 to match the class indices (1-4)


# Example usage:
image_path = r"C:\Users\MSI\Downloads\IIT STUFF\CM 2603 DS\CW implementation testing\DATASETS\test\4\9215922R.png"
predicted_grade = predict_grade(image_path)
print(f"Predicted Grade: {predicted_grade}")
