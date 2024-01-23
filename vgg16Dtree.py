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
import cv2
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.src.callbacks import LearningRateScheduler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import tensorflow as tf
from keras import layers, models, optimizers, regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout


# Function to load and preprocess images
def load_and_preprocess_data(folder_path):
    images = []
    labels = []

    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)

        for file_name in os.listdir(label_path):
            img_path = os.path.join(label_path, file_name)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # Use color images for VGG16
            img = cv2.resize(img, (224, 224))  # Resize for VGG16 input size
            images.append(img)
            labels.append(label)

    return np.array(images), np.array(labels)


# Load and preprocess training data
train_path = r"C:\Users\MSI\Downloads\IIT STUFF\CM 2603 DS\CW implementation testing\DATASETS\train"
X_train, y_train = load_and_preprocess_data(train_path)

# Load and preprocess validation data
validate_path = r"C:\Users\MSI\Downloads\IIT STUFF\CM 2603 DS\CW implementation testing\DATASETS\val"
X_val, y_val = load_and_preprocess_data(validate_path)

# Convert labels to numerical values
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_val = label_encoder.transform(y_val)

# Preprocess images for VGG16
X_train = preprocess_input(X_train)
X_val = preprocess_input(X_val)

# Load pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Extract features using VGG16
train_features = base_model.predict(X_train)
val_features = base_model.predict(X_val)

# Flatten the features
train_features_flat = train_features.reshape(train_features.shape[0], -1)
val_features_flat = val_features.reshape(val_features.shape[0], -1)

# Build a model on top of VGG16 features
model = Sequential()
model.add(Dense(256, activation='relu', input_dim=train_features_flat.shape[1]))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))  # Assuming 4 classes (KL grades 1-4)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_features_flat, y_train, epochs=24, batch_size=32, validation_data=(val_features_flat, y_val))

# Evaluate the model on the validation set
y_pred = np.argmax(model.predict(val_features_flat), axis=1)
accuracy = accuracy_score(y_val, y_pred)
print("Validation Accuracy:", accuracy)
