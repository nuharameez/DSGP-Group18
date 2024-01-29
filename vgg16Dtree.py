# import os
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.model_selection import GridSearchCV
# from xgboost import XGBClassifier
# from sklearn.metrics import classification_report
# from keras.preprocessing import image
# from keras.applications.vgg16 import VGG16, preprocess_input
# from keras.models import Model
#
# # Function to load and preprocess images
# def load_and_preprocess(data_folder):
#     # Lists to store image paths and corresponding labels
#     x_paths = []
#     y_labels = []
#
#     # Loop through each subdirectory (KL grade)
#     for kl_grade in os.listdir(data_folder):
#         kl_path = os.path.join(data_folder, kl_grade)
#
#         # Assuming each subdirectory contains image files
#         for img_file in os.listdir(kl_path):
#             img_path = os.path.join(kl_path, img_file)
#
#             # Read the KL grade from the subdirectory name
#             label = int(kl_grade.replace("KL", ""))
#
#             x_paths.append(img_path)
#             y_labels.append(label)
#
#     return x_paths, y_labels
#
# # Function to load and preprocess images
# def load_and_preprocess_images(image_paths):
#     images = []
#     for img_path in image_paths:
#         img = image.load_img(img_path, target_size=(224, 224))
#         img_array = image.img_to_array(img)
#         img_array = np.expand_dims(img_array, axis=0)
#         img_array = preprocess_input(img_array)
#         images.append(img_array)
#     return np.vstack(images)
#
# # Function to extract features using VGG16
# def extract_features(image_paths):
#     base_model = VGG16(weights='imagenet', include_top=False)
#     model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)  # Adjust layer as needed
#
#     preprocessed_images = load_and_preprocess_images(image_paths)
#     features = model.predict(preprocessed_images, verbose=1)
#     return features.reshape(features.shape[0], -1)
#
# # Load train and validate data
# train_folder = r"C:\Users\MSI\Downloads\IIT STUFF\CM 2603 DS\CW implementation testing\DATASETS\CustomTrain"
# validate_folder = r"C:\Users\MSI\Downloads\IIT STUFF\CM 2603 DS\CW implementation testing\DATASETS\CustomVal"
#
# train_images, train_labels = load_and_preprocess(train_folder)
# validate_images, validate_labels = load_and_preprocess(validate_folder)
#
# # Extract features (replace this with your feature extraction code)
# train_features = extract_features(train_images)
# validate_features = extract_features(validate_images)
#
# # Convert to numpy arrays and adjust labels to start from 0
# X_train = np.array(train_features)
# y_train = np.array(train_labels) - 1  # Adjust labels to start from 0
#
# X_validate = np.array(validate_features)
# y_validate = np.array(validate_labels) - 1  # Adjust labels to start from 0
#
# # Initialize and train XGBoost model
# model = XGBClassifier(max_depth=8, subsample=0.8, objective='multi:softmax', num_class=4)  # 4 classes for KL grades 0 to 3
# model.fit(X_train, y_train)
#
# # Validate the model
# predictions = model.predict(X_validate)
# accuracy = accuracy_score(y_validate, predictions)
#
# print("Validation Accuracy:", accuracy)
#
# # Print classification report
# print(classification_report(y_validate, predictions))

import os
# import cv2
# import numpy as np
# from keras.applications.vgg16 import VGG16, preprocess_input
# from keras.src.callbacks import LearningRateScheduler
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import accuracy_score
# import tensorflow as tf
# from keras import layers, models, optimizers, regularizers
# from keras.preprocessing.image import ImageDataGenerator
# from keras.models import Sequential
# from keras.layers import Dense, Flatten, Dropout
#
#
# # Function to load and preprocess images
# def load_and_preprocess_data(folder_path):
#     images = []
#     labels = []
#
#     for label in os.listdir(folder_path):
#         label_path = os.path.join(folder_path, label)
#
#         for file_name in os.listdir(label_path):
#             img_path = os.path.join(label_path, file_name)
#             img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # Use color images for VGG16
#             img = cv2.resize(img, (224, 224))  # Resize for VGG16 input size
#             images.append(img)
#             labels.append(label)
#
#     return np.array(images), np.array(labels)
#
#
# # Load and preprocess training data
# train_path = r"C:\Users\MSI\Downloads\IIT STUFF\CM 2603 DS\CW implementation testing\DATASETS\CustomTrain"
# X_train, y_train = load_and_preprocess_data(train_path)
#
# # Load and preprocess validation data
# validate_path = r"C:\Users\MSI\Downloads\IIT STUFF\CM 2603 DS\CW implementation testing\DATASETS\CustomVal"
# X_val, y_val = load_and_preprocess_data(validate_path)
#
# # Convert labels to numerical values
# label_encoder = LabelEncoder()
# y_train = label_encoder.fit_transform(y_train)
# y_val = label_encoder.transform(y_val)
#
# # Preprocess images for VGG16
# X_train = preprocess_input(X_train)
# X_val = preprocess_input(X_val)
#
# # Load pre-trained VGG16 model
# base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
#
# # Extract features using VGG16
# train_features = base_model.predict(X_train)
# val_features = base_model.predict(X_val)
#
# # Flatten the features
# train_features_flat = train_features.reshape(train_features.shape[0], -1)
# val_features_flat = val_features.reshape(val_features.shape[0], -1)
#
# # Build a model on top of VGG16 features
# model = Sequential()
# model.add(Dense(256, activation='relu', input_dim=train_features_flat.shape[1]))
# model.add(Dropout(0.5))
# model.add(Dense(4, activation='softmax'))  # Assuming 4 classes (KL grades 1-4)
#
# # Compile the model
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#
# # Train the model
# model.fit(train_features_flat, y_train, epochs=24, batch_size=32, validation_data=(val_features_flat, y_val))
#
# # Evaluate the model on the validation set
# y_pred = np.argmax(model.predict(val_features_flat), axis=1)
# accuracy = accuracy_score(y_val, y_pred)
# print("Validation Accuracy:", accuracy)

import os
import cv2
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Input
from keras import models
import joblib

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# Function to load and preprocess images
def load_and_preprocess_data(folder_path):
    images = []
    labels = []

    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)

        for file_name in os.listdir(label_path):
            img_path = os.path.join(label_path, file_name)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (224, 224))
            images.append(img)
            labels.append(label)

    return np.array(images), np.array(labels)


# Load and preprocess data
data_path = r"C:\Users\MSI\Downloads\IIT STUFF\CM 2603 DS\CW implementation testing\DATASETS\CustomTrain"
X, y = load_and_preprocess_data(data_path)

# Convert labels to numerical values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Preprocess images for VGG16
X = preprocess_input(X)

# Load pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Extract features using VGG16
features = base_model.predict(X)
features_flat = features.reshape(features.shape[0], -1)

# Define K-fold cross-validation
kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

# Initialize variables to store performance metrics
accuracy_scores = []

# Build the model outside the loop
input_layer = Input(shape=(features_flat.shape[1],))
x = Dense(256, activation='relu')(input_layer)
x = Dropout(0.5)(x)
output_layer = Dense(4, activation='softmax')(x)  # Assuming 4 classes (KL grades 1-4)

model = models.Model(inputs=input_layer, outputs=output_layer)

# Compile the model outside the loop
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Iterate over K folds
for train_index, val_index in kfold.split(features_flat, y):
    X_train, X_val = features_flat[train_index], features_flat[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # Train the model
    model.fit(X_train, y_train, epochs=40, batch_size=32, validation_data=(X_val, y_val))

    # Evaluate the model on the validation set
    y_pred = np.argmax(model.predict(X_val), axis=1)
    accuracy = accuracy_score(y_val, y_pred)
    accuracy_scores.append(accuracy)

# Print average accuracy across all folds
print("Average Validation Accuracy:", np.mean(accuracy_scores))

# Save the trained model
model.save('VGG16KFold.h5')

# Save the label encoder
joblib.dump(label_encoder, 'label_encoder_VGG16KFold.joblib')

# Function to load and preprocess test data
# def load_and_preprocess_test_data(folder_path):
#     images = []
#     labels = []
#
#     for label in os.listdir(folder_path):
#         label_path = os.path.join(folder_path, label)
#
#         for file_name in os.listdir(label_path):
#             img_path = os.path.join(label_path, file_name)
#             img = cv2.imread(img_path, cv2.IMREAD_COLOR)
#             img = cv2.resize(img, (224, 224))
#             images.append(img)
#             labels.append(label)
#
#     return np.array(images), np.array(labels)
#
#
# # Load and preprocess test data
# test_path = r"C:\Users\MSI\Downloads\IIT STUFF\CM 2603 DS\CW implementation testing\DATASETS\CustomTest"
# X_test, y_test = load_and_preprocess_test_data(test_path)
#
# # Convert test labels to numerical values
# y_test = label_encoder.transform(y_test)
#
# # Preprocess test images for VGG16
# X_test = preprocess_input(X_test)
#
# # Extract features using VGG16
# test_features = base_model.predict(X_test)
# test_features_flat = test_features.reshape(test_features.shape[0], -1)
#
# # Evaluate the model on the test set
# y_pred_test = np.argmax(model.predict(test_features_flat), axis=1)
#
# # Display actual and predicted grades for each image, separated by folder grades
# unique_labels = np.unique(y_test)
# correct_predictions_count = 0
#
# for label in unique_labels:
#     folder_indices = np.where(y_test == label)[0]
#     folder_correct_predictions = 0
#
#     print(f"\nFolder Grade {label+1}:")
#     for i in folder_indices:
#         actual_grade = label_encoder.inverse_transform([y_test[i]])[0]
#         predicted_grade = label_encoder.inverse_transform([y_pred_test[i]])[0]
#
#         if actual_grade == predicted_grade:
#             folder_correct_predictions += 1
#             correct_predictions_count += 1
#
#         print(f"Image {i + 1}: Actual Grade = {actual_grade}, Predicted Grade = {predicted_grade}")
#
#     folder_accuracy = folder_correct_predictions / len(folder_indices) * 100
#     print(f"Folder Accuracy: {folder_accuracy:.2f}%")
#
# # Calculate and print overall accuracy
# overall_accuracy = correct_predictions_count / len(y_test) * 100
# print(f"\nOverall Accuracy: {overall_accuracy:.2f}%")


# Function to preprocess a single image and make predictions

