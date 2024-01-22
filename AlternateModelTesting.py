# import os
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# from sklearn.metrics import accuracy_score, classification_report
# import xgboost as xgb
# from sklearn.preprocessing import LabelEncoder
# from skimage import io, transform
# import joblib
#
# # Set the path to your dataset
# dataset_path = r'C:\Users\MSI\Downloads\IIT STUFF\CM 2603 DS\CW implementation testing\DATASETS'
#
#
# # Function to load and preprocess images
# def load_images(folder_path, label):
#     images = []
#     labels = []
#     for filename in tqdm(os.listdir(folder_path)):
#         if filename.endswith('.png'):
#             img_path = os.path.join(folder_path, filename)
#             img = io.imread(img_path)
#             img = transform.resize(img, (224, 224))  # Resize the image to 224x224
#             images.append(img.flatten())  # Flatten the image array
#             labels.append(label)
#     return np.array(images), np.array(labels)
#
# # Load and preprocess test data
# test_data = []
# test_labels = []
# for kl_grade in range(1, 5):
#     kl_folder_path = os.path.join(dataset_path, 'CustomTest', str(kl_grade))
#     images, labels = load_images(kl_folder_path, kl_grade)
#     test_data.extend(images)
#     test_labels.extend(labels)
#
# # Convert data to numpy arrays
# X_test = np.array(test_data)
# y_test = np.array(test_labels)
#
# # Load the saved model
# model_filename = 'xgboost_model2.joblib'
# loaded_model = joblib.load(model_filename)
#
# # Load the LabelEncoder fitted during training
# label_encoder_filename = 'label_encoder2.joblib'
# label_encoder = joblib.load(label_encoder_filename)
#
# # Ensure that the LabelEncoder has the same classes as used during training
# new_classes = set(label_encoder.classes_) | set(y_test)
# label_encoder.classes_ = np.array(list(new_classes))
#
# # Make predictions on the test set using the loaded model
# y_pred_test = loaded_model.predict(X_test)
#
# # Convert class names to strings
# class_names = list(map(str, label_encoder.classes_))
#
# # Display predictions for each folder
# unique_folders = np.unique(y_test)
# for folder in unique_folders:
#     folder_indices = np.where(y_test == folder)[0]
#     folder_images = X_test[folder_indices]
#     folder_labels = y_test[folder_indices]
#
#     # Make predictions for images in the current folder
#     folder_predictions = loaded_model.predict(folder_images)
#
#     print(f"\nPredictions for Folder {folder}:")
#
#     # Adjust actual labels to match 0-indexed format
#     adjusted_actual_labels = folder_labels - 1
#
#     # Optionally, print some actual and predicted labels for individual samples
#     for i in range(min(20, len(folder_images))):
#         image_path = os.path.join(dataset_path, 'CustomTest', str(folder), f'image_{i + 1}.png')
#
#         try:
#             true_label = int(label_encoder.inverse_transform([folder_labels[i]])[0])
#         except ValueError:
#             true_label = -1  # Handle the case of an unseen label
#
#         predicted_label = int(label_encoder.inverse_transform([folder_predictions[i]])[0])
#         print(
#             f"Actual: {true_label - 1 if true_label != -1 else '4'}, Predicted: {predicted_label}")
#
# # Evaluate overall performance
# encoded_labels_test = label_encoder.transform(y_test)
# y_pred_probabilities = loaded_model.predict_proba(X_test)
#
# # Get the predicted class labels by finding the index with maximum probability
# y_pred_labels = np.argmax(y_pred_probabilities, axis=1)
#
# print(classification_report(encoded_labels_test, y_pred_labels, target_names=class_names))

import cv2
from skimage.transform import resize
import os
from skimage import io
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# Function for making all images square
def square(img):
    diff = img.shape[0] - img.shape[1]
    if diff % 2 == 0:
        pad1 = int(np.floor(np.abs(diff) / 2))
        pad2 = int(np.floor(np.abs(diff) / 2))
    else:
        pad1 = int(np.floor(np.abs(diff) / 2))
        pad2 = int(np.floor(np.abs(diff) / 2)) + 1

    if diff == 0:
        return img
    elif diff > 0:
        return np.pad(img, [(0, 0), (pad1, pad2)], 'constant', constant_values=(0))
    elif diff < 0:
        return np.pad(img, [(pad1, pad2), (0, 0)], 'constant', constant_values=(0))


def preprocess(img):
    blurred_image = cv2.GaussianBlur(img, (5, 5), 0)

    # Example: Histogram equalization for contrast enhancement
    equalized_image = cv2.equalizeHist(blurred_image)

    # Thresholding
    _, thresholded_image = cv2.threshold(equalized_image, 150, 255, cv2.THRESH_BINARY)

    return thresholded_image


# Function for resizing and cropping
def resize_crop(img):
    img = resize(img, (224, 224))
    img = img[:, 14:115]
    return img.flatten()

def load_images(folder_path, label):
    images = []
    labels = []
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith('.png'):
            img_path = os.path.join(folder_path, filename)
            img = io.imread(img_path)
            # img = square(img)
            img = preprocess(img)
            img = resize_crop(img)
            images.append(img)  # No need to flatten here
            labels.append(label)
    return np.array(images), np.array(labels)

# Set the path to your dataset
dataset_path = r'C:\Users\MSI\Downloads\IIT STUFF\CM 2603 DS\CW implementation testing\DATASETS'

# Load and preprocess test data
test_data = []
test_labels = []
for kl_grade in range(1, 5):
    kl_folder_path = os.path.join(dataset_path, 'CustomTest', str(kl_grade))  # Assuming 'test' folder for testing data
    images, labels = load_images(kl_folder_path, kl_grade)
    test_data.extend(images)
    test_labels.extend(labels)

# Convert data to numpy arrays
X_test = np.array(test_data)
y_test = np.array(test_labels)

# Load the pre-trained model
model_filename = 'xgboost_model3.joblib'
model = joblib.load(model_filename)

# Load the label encoder
label_encoder_filename = 'label_encoder3.joblib'
label_encoder = joblib.load(label_encoder_filename)

# Convert labels to numerical values
encoded_labels_test = label_encoder.transform(y_test)

# Make predictions on the test set
y_pred_test = model.predict(X_test)

# Display predictions for each folder
unique_folders = np.unique(y_test)
for folder in unique_folders:
    folder_indices = np.where(y_test == folder)[0]
    folder_predictions = y_pred_test[folder_indices]

    print(f"\nPredictions for Folder {folder}:")
    print(f"Actual Labels: {y_test[folder_indices]}")
    print(f"Predicted Labels: {folder_predictions}")

    # Optionally, print some actual and predicted labels for individual samples
    for actual, predicted in zip(y_test[folder_indices][:10], folder_predictions[:10]):
        print(f"Actual: {actual}, Predicted: {predicted + 1}")

# Evaluate the model on the test set
accuracy_test = accuracy_score(encoded_labels_test, y_pred_test)
print(f'Test Accuracy: {accuracy_test}')

# Convert class names to strings
class_names = list(map(str, label_encoder.classes_))

# Print classification report
print(classification_report(encoded_labels_test, y_pred_test, target_names=class_names))
