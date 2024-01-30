import os
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

# Load the saved model
loaded_model = load_model('knee_bone_identifier.h5')

# Directory containing the test images
test_images_dir = 'Bone_Xrays/Test/'

# Loop through each file in the directory
for filename in os.listdir(test_images_dir):
        # Construct the full file path
        img_path = os.path.join(test_images_dir, filename)

        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Rescale to match the training data

        # Apply the same preprocessing used during training
        #img_array = preprocess_input(img_array)


        # Make predictions using the loaded model
        predictions = loaded_model.predict(img_array)

        # Display the result
        if predictions[0][0] > 0.5:
            print(f"{filename}: Predicted - Knee Bone")
        else:
            print(f"{filename}: Predicted - Non-Knee Bone")
"""""
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the trained model
model = tf.keras.models.load_model('knee_bone_identifier.h5')

# Define the input shape of your images
input_shape = (224, 224, 3)  # Adjust the dimensions based on your images

# Set up the test data generator
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    'Bone_Xrays/Test',  # Path to your test folder
    target_size=input_shape[:2],
    batch_size=32,
    class_mode='binary',  # Binary classification
    classes=['other bones', 'knee bone']  # Class labels
)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_generator, verbose=2)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
"""""
