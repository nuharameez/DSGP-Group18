import os
import cv2
import numpy as np
from joblib import load
from sklearn.metrics import accuracy_score, classification_report
from keras.preprocessing.image import img_to_array, load_img


# Function to load and preprocess a single image
def load_and_preprocess(data_folder):
    # Lists to store image paths and corresponding labels
    x_paths = []
    y_labels = []

    # Loop through each subdirectory (KL grade)
    for kl_grade in os.listdir(data_folder):
        kl_path = os.path.join(data_folder, kl_grade)

        # Assuming each subdirectory contains image files
        for img_file in os.listdir(kl_path):
            img_path = os.path.join(kl_path, img_file)

            # Read the KL grade from the subdirectory name
            label = int(kl_grade)

            x_paths.append(img_path)
            y_labels.append(label)

    # Load images and labels
    X = []
    y = []

    for img_path, label in zip(x_paths, y_labels):
        print(f"Image Path: {img_path}, Label: {label}")
        # Read and preprocess the image
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = cv2.cvtColor(np.array(img_array, dtype=np.uint8), cv2.COLOR_BGR2RGB)
        img_array = cv2.resize(img_array, (224, 224))
        img_array = cv2.GaussianBlur(img_array, (5, 5), 0)
        img_array = img_array / 255.0  # Normalize pixel values to [0, 1]

        # Append the image and label to the lists
        X.append(img_array)
        y.append(label)

    return np.array(X), np.array(y)


# Load the saved decision tree model
model_filename = "decision_tree_model2.joblib"
loaded_model = load(model_filename)

# Specify the path to the folder containing KL grade subfolders
test_data_folder = r"C:\Users\MSI\Downloads\IIT STUFF\CM 2603 DS\CW implementation testing\DATASETS\CustomTest"

X_test, y_test = load_and_preprocess(test_data_folder)

X_test_flattened = X_test.reshape(X_test.shape[0], -1)

y_predict = loaded_model.predict(X_test_flattened)

accuracy = accuracy_score(y_test, y_predict)
print(f"Accuracy: {accuracy:.2f}")

print("Classification Report:")
print(classification_report(y_test, y_predict, zero_division=1))

# Display predictions for each folder
unique_folders = np.unique(y_test)
for folder in unique_folders:
    folder_indices = np.where(y_test == folder)[0]
    folder_predictions = y_predict[folder_indices]

    print(f"\nPredictions for Folder {folder}:")
    print(f"Actual Labels: {y_test[folder_indices]}")
    print(f"Predicted Labels: {folder_predictions}")

    # Optionally, print some actual and predicted labels for individual samples
    for actual, predicted in zip(y_test[folder_indices][:10], folder_predictions[:10]):
        print(f"Actual: {actual}, Predicted: {predicted}")
