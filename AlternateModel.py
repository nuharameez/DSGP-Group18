# import os
# import numpy as np
# import tensorflow as tf
# from keras.preprocessing.image import ImageDataGenerator
# from keras.applications import ResNet50
# from keras import layers, models
# from sklearn.metrics import classification_report, accuracy_score
#
# # Define paths to your dataset folders
# train_data_dir = r'C:\Users\MSI\Downloads\IIT STUFF\CM 2603 DS\CW implementation testing\DATASETS\CustomTrain'
# validation_data_dir = r'C:\Users\MSI\Downloads\IIT STUFF\CM 2603 DS\CW implementation testing\DATASETS\CustomVal'
# test_data_dir = r'C:\Users\MSI\Downloads\IIT STUFF\CM 2603 DS\CW implementation testing\DATASETS\CustomTest'
#
# # Constants
# img_height, img_width = 224, 224
# batch_size = 32
#
# # Data Augmentation for training images
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True
# )
#
# # Data Augmentation for validation and test images
# val_test_datagen = ImageDataGenerator(rescale=1./255)
#
# # Load and configure the ResNet50 model
# base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
#
# # Freeze the layers in the base model
# for layer in base_model.layers:
#     layer.trainable = False
#
# # Create a new model using the base model
# model = models.Sequential()
# model.add(base_model)
# model.add(layers.GlobalAveragePooling2D())
# model.add(layers.Dense(128, activation='relu'))
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(4, activation='softmax'))
#
# # Compile the model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#
# # Create data generators
# train_generator = train_datagen.flow_from_directory(
#     train_data_dir,
#     target_size=(img_height, img_width),
#     batch_size=batch_size,
#     class_mode='categorical'
# )
#
# validation_generator = val_test_datagen.flow_from_directory(
#     validation_data_dir,
#     target_size=(img_height, img_width),
#     batch_size=batch_size,
#     class_mode='categorical'
# )
#
# # Train the model
# history = model.fit(
#     train_generator,
#     steps_per_epoch=train_generator.samples // batch_size,
#     epochs=10,  # Increase the number of epochs as needed
#     validation_data=validation_generator,
#     validation_steps=validation_generator.samples // batch_size
# )
#
# # Evaluate the model on the test set
# test_generator = val_test_datagen.flow_from_directory(
#     test_data_dir,
#     target_size=(img_height, img_width),
#     batch_size=batch_size,
#     class_mode='categorical',
#     shuffle=False
# )
#
# y_true = test_generator.classes
# y_pred = model.predict(test_generator)
# y_pred_classes = np.argmax(y_pred, axis=1)
#
# # Calculate accuracy and print classification report
# accuracy = accuracy_score(y_true, y_pred_classes)
# classification_rep = classification_report(y_true, y_pred_classes)
#
# print(f"Accuracy: {accuracy}")
# print(f"Classification Report:\n{classification_rep}")

# import os
# import numpy as np
# from keras.preprocessing.image import ImageDataGenerator
# from keras import layers
# from keras import models
# from keras import optimizers
# from keras.src.applications import EfficientNetB0
# from sklearn.metrics import classification_report, accuracy_score
#
# # Set the paths to your data
# train_dir = r'C:\Users\MSI\Downloads\IIT STUFF\CM 2603 DS\CW implementation testing\DATASETS\CustomTrain'
# validation_dir = r'C:\Users\MSI\Downloads\IIT STUFF\CM 2603 DS\CW implementation testing\DATASETS\CustomVal'
#
# # Parameters
# img_size = (224, 224)
# batch_size = 32
#
# # Data augmentation for training
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )
#
# # Validation data should not be augmented
# validation_datagen = ImageDataGenerator(rescale=1./255)
#
# # Load data from directories
# train_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=img_size,
#     batch_size=batch_size,
#     class_mode='categorical'
# )
#
# validation_generator = validation_datagen.flow_from_directory(
#     validation_dir,
#     target_size=img_size,
#     batch_size=batch_size,
#     class_mode='categorical'
# )
#
# # Build the model with EfficientNetB0 as the convolutional base
# conv_base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
#
# model = models.Sequential()
# model.add(conv_base)
# model.add(layers.GlobalAveragePooling2D())
# model.add(layers.Dense(512, activation='relu'))
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(4, activation='softmax'))  # 4 classes for KL grades 1 to 4
#
# # Freeze the convolutional base
# conv_base.trainable = False
#
# # Compile the model
# model.compile(loss='categorical_crossentropy',
#               optimizer=optimizers.Adam(lr=1e-4),
#               metrics=['accuracy'])
#
# # Train the model
# history = model.fit(
#     train_generator,
#     steps_per_epoch=train_generator.samples // batch_size,
#     epochs=20,  # You may need to adjust the number of epochs based on your dataset and computational resources
#     validation_data=validation_generator,
#     validation_steps=validation_generator.samples // batch_size
# )
#
# # Evaluate the model on the test set
# test_dir = r'C:\Users\MSI\Downloads\IIT STUFF\CM 2603 DS\CW implementation testing\DATASETS\CustomTest'
#
# test_datagen = ImageDataGenerator(rescale=1./255)
#
# test_generator = test_datagen.flow_from_directory(
#     test_dir,
#     target_size=img_size,
#     batch_size=batch_size,
#     class_mode='categorical',
#     shuffle=False
# )
#
# # Predict on the test set
# predictions = model.predict(test_generator)
# predicted_labels = np.argmax(predictions, axis=1)
#
# true_labels = test_generator.classes
#
# # Decode class labels
# class_labels = list(test_generator.class_indices.keys())
#
# # Evaluate the model
# accuracy = accuracy_score(true_labels, predicted_labels)
# print("Accuracy: {:.2f}%".format(accuracy * 100))
#
# classification_report_result = classification_report(true_labels, predicted_labels, target_names=class_labels)
# print("Classification Report:\n", classification_report_result)

import os

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from skimage import io, transform
from sklearn.metrics import classification_report

# Set the path to your dataset
dataset_path = r'C:\Users\MSI\Downloads\IIT STUFF\CM 2603 DS\CW implementation testing\DATASETS'


# Function to load and preprocess images
def load_images(folder_path, label):
    images = []
    labels = []
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith('.png'):
            img_path = os.path.join(folder_path, filename)
            img = io.imread(img_path)
            img = transform.resize(img, (224, 224))  # Resize the image to 224x224
            images.append(img.flatten())  # Flatten the image array
            labels.append(label)
    return np.array(images), np.array(labels)


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



# Create an XGBoost model
model = xgb.XGBClassifier(max_depth=6, subsample=0.8, objective='multi:softmax', num_class=len(np.unique(encoded_labels_train)))

# Train the model
model.fit(X_train, encoded_labels_train)

# Save the trained model to a file
model_filename = 'xgboost_model2.joblib'
joblib.dump(model, model_filename)
print(f"Model saved as '{model_filename}'")

label_encoder_filename = 'label_encoder2.joblib'
joblib.dump(label_encoder, label_encoder_filename)
print(f"LabelEncoder saved as '{label_encoder_filename}'")

# Make predictions on the validate set
y_pred_validate = model.predict(X_validate)

# Evaluate the model on validate set
accuracy_validate = accuracy_score(encoded_labels_validate, y_pred_validate)
print(f'Validation Accuracy: {accuracy_validate}')

# Convert class names to strings
class_names = list(map(str, label_encoder.classes_))

# Print classification report
print(classification_report(encoded_labels_validate, y_pred_validate, target_names=class_names))
