# import os
# import cv2
# import numpy as np
# from skimage import feature
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import accuracy_score, classification_report
# from keras.applications.vgg16 import VGG16, preprocess_input
# from keras.preprocessing.image import load_img, img_to_array
#
# # Load the pre-trained VGG16 model (excluding the top layer)
# vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
#
#
# # Function for image preprocessing using VGG16 standards
# def preprocess_image_vgg16(image_path):
#     img = load_img(image_path, target_size=(224, 224))
#     img_array = img_to_array(img)
#     img_array = cv2.cvtColor(np.array(img_array, dtype=np.uint8), cv2.COLOR_BGR2RGB)
#     img_array = cv2.resize(img_array, (224, 224))
#     img_array = cv2.GaussianBlur(img_array, (5, 5), 0)
#     img_array = preprocess_input(img_array)  # VGG16 preprocessing
#     return img_array
#
#
# # Function for feature extraction using VGG16 model
# def extract_features_vgg16(image):
#     features = vgg16_model.predict(np.expand_dims(image, axis=0))
#     flattened_features = features.flatten()
#     return flattened_features
#
#
# # Load dataset
# data_folder = r"C:\Users\MSI\Downloads\IIT STUFF\CM 2603 DS\CW implementation testing\DATASETS"  # Change this to the actual path
# kl_grades = ["1", "2", "3", "4"]
#
# image_paths = []
# labels = []
#
# for grade in kl_grades:
#     grade_folder = os.path.join(data_folder, "CustomTrain", grade)
#     images = os.listdir(grade_folder)
#
#     for image_name in images:
#         image_path = os.path.join(grade_folder, image_name)
#         image_paths.append(image_path)
#         labels.append(int(grade))
#
# # Preprocess images using VGG16 standards and extract features
# processed_images = [preprocess_image_vgg16(image_path) for image_path in image_paths]
# features_vgg16 = [extract_features_vgg16(image) for image in processed_images]
#
# # Split the dataset into training and validation sets
# X_train, X_val, y_train, y_val = train_test_split(features_vgg16, labels, test_size=0.2, random_state=42)
#
# # Hyperparameter tuning using GridSearchCV
# param_grid = {'max_depth': [None, 10, 20, 30],
#               'min_samples_split': [2, 5, 10],
#               'min_samples_leaf': [1, 2, 4]}
#
# clf = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5)
# clf.fit(X_train, y_train)
#
# # Print the best parameters found by GridSearchCV
# print("Best Parameters:", clf.best_params_)
#
# # Make predictions on the validation set
# val_predictions = clf.predict(X_val)
#
# # Evaluate the model on the validation set
# accuracy = accuracy_score(y_val, val_predictions)
# print(f"Validation Accuracy: {accuracy}")
# print("Classification Report:")
# print(classification_report(y_val, val_predictions))

# import os
# import joblib
# import numpy as np
# from tqdm import tqdm
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report
# import xgboost as xgb
# from sklearn.preprocessing import LabelEncoder
# from skimage import io, transform
# from keras.applications import VGG16
# from keras.applications.vgg16 import preprocess_input
#
# from keras.applications.vgg16 import VGG16, preprocess_input
#
# # Load VGG16 model with weights pre-trained on ImageNet
# vgg16_model = VGG16(weights='imagenet', include_top=False)
#
# # Function to load and preprocess images using VGG16 as a feature extractor
# def load_images_with_feature_extraction(folder_path, label, model):
#     images = []
#     labels = []
#     for filename in tqdm(os.listdir(folder_path)):
#         if filename.endswith('.png'):
#             img_path = os.path.join(folder_path, filename)
#             img = io.imread(img_path)
#             img = transform.resize(img, (224, 224, 3))  # Ensure the input shape is (224, 224, 3)
#             img = preprocess_input(img)  # Preprocess for VGG16
#             img = np.expand_dims(img, axis=0)  # Expand dimensions to include batch size
#             img_features = model.predict(img).flatten()  # Extract features
#             images.append(img_features)
#             labels.append(label)
#     return np.array(images), np.array(labels)
#
#
# # Set the path to your dataset
# dataset_path = r'C:\Users\MSI\Downloads\IIT STUFF\CM 2603 DS\CW implementation testing\DATASETS'
#
# # Load and preprocess train data with VGG16 feature extraction
# train_data = []
# train_labels = []
# for kl_grade in range(1, 5):
#     kl_folder_path = os.path.join(dataset_path, 'CustomTrain', str(kl_grade))
#     images, labels = load_images_with_feature_extraction(kl_folder_path, kl_grade, vgg16_model)
#     train_data.extend(images)
#     train_labels.extend(labels)
#
# # Load and preprocess validate data with VGG16 feature extraction
# validate_data = []
# validate_labels = []
# for kl_grade in range(1, 5):
#     kl_folder_path = os.path.join(dataset_path, 'CustomVal', str(kl_grade))
#     images, labels = load_images_with_feature_extraction(kl_folder_path, kl_grade, vgg16_model)
#     validate_data.extend(images)
#     validate_labels.extend(labels)
#
# # Convert data to numpy arrays
# X_train = np.array(train_data)
# y_train = np.array(train_labels)
# X_validate = np.array(validate_data)
# y_validate = np.array(validate_labels)
#
# # Convert labels to numerical values
# label_encoder = LabelEncoder()
# encoded_labels_train = label_encoder.fit_transform(y_train)
# encoded_labels_validate = label_encoder.transform(y_validate)
#
# # Create an XGBoost model
# model = xgb.XGBClassifier(max_depth=6, subsample=0.8, objective='multi:softmax', num_class=len(np.unique(encoded_labels_train)))
#
# # Train the model
# model.fit(X_train, encoded_labels_train)
#
# # Save the trained model to a file
# # model_filename = 'xgboost_model_with_feature_extraction.joblib'
# # joblib.dump(model, model_filename)
# # print(f"Model saved as '{model_filename}'")
# #
# # label_encoder_filename = 'label_encoder_with_feature_extraction.joblib'
# # joblib.dump(label_encoder, label_encoder_filename)
# # print(f"LabelEncoder saved as '{label_encoder_filename}'")
#
# # Make predictions on the validate set
# y_pred_validate = model.predict(X_validate)
#
# # Evaluate the model on validate set
# accuracy_validate = accuracy_score(encoded_labels_validate, y_pred_validate)
# print(f'Validation Accuracy: {accuracy_validate}')
#
# # Convert class names to strings
# class_names = list(map(str, label_encoder.classes_))
#
# # Print classification report
# print(classification_report(encoded_labels_validate, y_pred_validate, target_names=class_names))
#

import skimage.filters as filters
from skimage.transform import resize
import os
from skimage import exposure
import joblib
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.morphology import reconstruction
from skimage.util import img_as_ubyte, img_as_float
from tqdm import tqdm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from skimage import io, transform

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

# Function for enhancing the contrast
def contrast(img):
    img = img_as_ubyte(img / 255)
    img = img_as_float(img)
    img = gaussian_filter(img, 1)
    h = 0.8
    seed = img - h
    mask = img
    dilated = reconstruction(seed, mask, method='dilation')
    img_dil_adapteq = exposure.equalize_adapthist(img - dilated, clip_limit=0.03)
    return img_dil_adapteq

# Function for thresholding
def threshold(img):
    img_threshold = filters.threshold_li(img)
    img_new = np.ones(img.shape[:2], dtype="float")
    img_new[(img < img_threshold) | (img > 250)] = 0
    return img_new

# Function for resizing and cropping
def resize_crop(img):
    img = resize(img, (224, 224))
    img = img[:, 14:115]
    return img.flatten()

# Function to load and preprocess images
def load_images(folder_path, label):
    images = []
    labels = []
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith('.png'):
            img_path = os.path.join(folder_path, filename)
            img = io.imread(img_path)
            img = square(img)
            img = contrast(img)
            img = threshold(img)
            img = resize_crop(img)
            images.append(img)  # No need to flatten here
            labels.append(label)
    return np.array(images), np.array(labels)

# Set the path to your dataset
dataset_path = r'C:\Users\MSI\Downloads\IIT STUFF\CM 2603 DS\CW implementation testing\DATASETS'

# Load and preprocess train data
train_data = []
train_labels = []
for kl_grade in range(1, 5):
    kl_folder_path = os.path.join(dataset_path, 'CustomTrain', str(kl_grade))
    images, labels = load_images(kl_folder_path, kl_grade)
    train_data.extend(images)
    train_labels.extend(labels)

# Load and preprocess validate data
validate_data = []
validate_labels = []
for kl_grade in range(1, 5):
    kl_folder_path = os.path.join(dataset_path, 'CustomVal', str(kl_grade))
    images, labels = load_images(kl_folder_path, kl_grade)
    validate_data.extend(images)
    validate_labels.extend(labels)

# Convert data to numpy arrays
X_train = np.array(train_data)
y_train = np.array(train_labels)
X_validate = np.array(validate_data)
y_validate = np.array(validate_labels)

# Convert labels to numerical values
label_encoder = LabelEncoder()
encoded_labels_train = label_encoder.fit_transform(y_train)
encoded_labels_validate = label_encoder.transform(y_validate)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
}

grid_search = GridSearchCV(xgb.XGBClassifier(objective='multi:softmax', num_class=len(np.unique(encoded_labels_train))),
                           param_grid=param_grid, cv=3, verbose=2)
grid_search.fit(X_train, encoded_labels_train)

# Get the best model from the grid search
best_model = grid_search.best_estimator_

# Save the trained model to a file
# model_filename = 'xgboost_model_optimized.joblib'
# joblib.dump(best_model, model_filename)
# print(f"Optimized Model saved as '{model_filename}'")

# Make predictions on the validate set
y_pred_validate = best_model.predict(X_validate)

# Evaluate the model on validate set
accuracy_validate = accuracy_score(encoded_labels_validate, y_pred_validate)
print(f'Validation Accuracy: {accuracy_validate}')

# Convert class names to strings
class_names = list(map(str, label_encoder.classes_))

# Print classification report
print(classification_report(encoded_labels_validate, y_pred_validate, target_names=class_names))