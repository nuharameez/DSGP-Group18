# import cv2
# import skimage.filters as filters
# from skimage.transform import resize
# import os
# from skimage import exposure
# import joblib
# import numpy as np
# from scipy.ndimage import gaussian_filter
# from skimage.morphology import reconstruction
# from skimage.util import img_as_ubyte, img_as_float
# from tqdm import tqdm
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import accuracy_score, classification_report
# import xgboost as xgb
# from sklearn.preprocessing import LabelEncoder
# from skimage import io, transform
#
#
# # Function for making all images square
# def square(img):
#     diff = img.shape[0] - img.shape[1]
#     if diff % 2 == 0:
#         pad1 = int(np.floor(np.abs(diff) / 2))
#         pad2 = int(np.floor(np.abs(diff) / 2))
#     else:
#         pad1 = int(np.floor(np.abs(diff) / 2))
#         pad2 = int(np.floor(np.abs(diff) / 2)) + 1
#
#     if diff == 0:
#         return img
#     elif diff > 0:
#         return np.pad(img, [(0, 0), (pad1, pad2)], 'constant', constant_values=(0))
#     elif diff < 0:
#         return np.pad(img, [(pad1, pad2), (0, 0)], 'constant', constant_values=(0))
#
#
# def preprocess(img):
#     blurred_image = cv2.GaussianBlur(img, (5, 5), 0)
#
#     # Example: Histogram equalization for contrast enhancement
#     equalized_image = cv2.equalizeHist(blurred_image)
#
#     # Thresholding
#     _, thresholded_image = cv2.threshold(equalized_image, 150, 255, cv2.THRESH_BINARY)
#
#     return thresholded_image
#
#
# # Function for resizing and cropping
# def resize_crop(img):
#     img = resize(img, (224, 224))
#     img = img[:, 14:115]
#     return img.flatten()
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
#             img = square(img)
#             img = preprocess(img)
#             img = resize_crop(img)
#             images.append(img)  # No need to flatten here
#             labels.append(label)
#     return np.array(images), np.array(labels)
#
#
# # Set the path to your dataset
# dataset_path = r'C:\Users\MSI\Downloads\IIT STUFF\CM 2603 DS\CW implementation testing\DATASETS'
#
# # Load and preprocess train data
# train_data = []
# train_labels = []
# for kl_grade in range(1, 5):
#     kl_folder_path = os.path.join(dataset_path, 'train', str(kl_grade))
#     images, labels = load_images(kl_folder_path, kl_grade)
#     train_data.extend(images)
#     train_labels.extend(labels)
#
# # Load and preprocess validate data
# validate_data = []
# validate_labels = []
# for kl_grade in range(1, 5):
#     kl_folder_path = os.path.join(dataset_path, 'val', str(kl_grade))
#     images, labels = load_images(kl_folder_path, kl_grade)
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
# # # Hyperparameter tuning with GridSearchCV
# # param_grid = {
# #     'max_depth': [4, 6, 8],
# #     'subsample': [0.7, 0.8, 0.9],
# #     'learning_rate': [0.01, 0.1, 0.3]
# # }
# #
# # grid_search = GridSearchCV(
# #     estimator=xgb.XGBClassifier(objective='multi:softmax', num_class=len(np.unique(encoded_labels_train))),
# #     param_grid=param_grid,
# #     scoring='accuracy',
# #     cv=3
# # )
# #
# # grid_search.fit(X_train, encoded_labels_train)
# #
# # best_params = grid_search.best_params_
# # best_model = grid_search.best_estimator_
# #
# # # Evaluate the best model on the validation set
# # y_pred_validate_tuned = best_model.predict(X_validate)
# # accuracy_validate_tuned = accuracy_score(encoded_labels_validate, y_pred_validate_tuned)
# # print(f'Validation Accuracy with Tuned Model: {accuracy_validate_tuned}')
# #
# # # Save the trained model to a file (if desired)
# # # model_filename = 'xgboost_tuned_model.joblib'
# # # joblib.dump(best_model, model_filename)
# # # print(f"Tuned Model saved as '{model_filename}'")
# #
# # # Save the label encoder (if desired)
# # # label_encoder_filename = 'label_encoder_tuned.joblib'
# # # joblib.dump(label_encoder, label_encoder_filename)
# # # print(f"LabelEncoder saved as '{label_encoder_filename}'")
# #
# # # Print best hyperparameters
# # print("Best Hyperparameters:")
# # print(best_params)
# #
# # # Print classification report
# # print("Classification Report:")
# # print(classification_report(encoded_labels_validate, y_pred_validate_tuned, target_names=list(map(str, label_encoder.classes_))))
#
# # Create an XGBoost model
# model = xgb.XGBClassifier(max_depth=8, subsample=0.8, objective='multi:softmax', num_class=len(np.unique(encoded_labels_train)))
#
# # Train the model
# model.fit(X_train, encoded_labels_train)
#
# # Save the trained model to a file
# model_filename = 'xgboost_model3.joblib'
# joblib.dump(model, model_filename)
# print(f"Model saved as '{model_filename}'")
#
# label_encoder_filename = 'label_encoder3.joblib'
# joblib.dump(label_encoder, label_encoder_filename)
# print(f"LabelEncoder saved as '{label_encoder_filename}'")
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

import os
import numpy as np
import cv2
from skimage import io
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
# from sklearn.externals import joblib
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from skimage.transform import resize
import xgboost as xgb

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

    # If the image is grayscale, convert it to a 3-channel image
    if img.ndim == 2:
        img = np.stack((img,) * 3, axis=-1)

    return img
# Function to load and preprocess images
def load_images_with_vgg16(folder_path, label, model):
    images = []
    labels = []
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith('.png'):
            img_path = os.path.join(folder_path, filename)
            img = io.imread(img_path)
            # img = square(img)
            # img = preprocess(img)
            img = resize_crop(img)
            #
            # Preprocess input for VGG16
            img = preprocess_input(img.reshape(1, 224, 224, 3))

            # Extract features using VGG16
            features = model.predict(img)

            images.append(features.flatten())
            labels.append(label)

    return np.array(images), np.array(labels)

# Load VGG16 model
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

# Set the path to your dataset
dataset_path = r'C:\Users\MSI\Downloads\IIT STUFF\CM 2603 DS\CW implementation testing\DATASETS'

# Load and preprocess train data
train_data = []
train_labels = []
for kl_grade in range(1, 5):
    kl_folder_path = os.path.join(dataset_path, 'CustomTrain', str(kl_grade))
    images, labels = load_images_with_vgg16(kl_folder_path, kl_grade, model)
    train_data.extend(images)
    train_labels.extend(labels)

# Load and preprocess validate data
validate_data = []
validate_labels = []
for kl_grade in range(1, 5):
    kl_folder_path = os.path.join(dataset_path, 'CustomVal', str(kl_grade))
    images, labels = load_images_with_vgg16(kl_folder_path, kl_grade, model)
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
model = xgb.XGBClassifier(max_depth=8, subsample=0.8, objective='multi:softmax', num_class=len(np.unique(encoded_labels_train)))

# Train the model
model.fit(X_train, encoded_labels_train)

# Save the trained model to a file
# model_filename = 'xgboost_model3.joblib'
# joblib.dump(model, model_filename)
# print(f"Model saved as '{model_filename}'")
#
# label_encoder_filename = 'label_encoder3.joblib'
# joblib.dump(label_encoder, label_encoder_filename)
# print(f"LabelEncoder saved as '{label_encoder_filename}'")

# Make predictions on the validate set
y_pred_validate = model.predict(X_validate)

# Evaluate the model on validate set
accuracy_validate = accuracy_score(encoded_labels_validate, y_pred_validate)
print(f'Validation Accuracy: {accuracy_validate}')

# Convert class names to strings
class_names = list(map(str, label_encoder.classes_))

# Print classification report
print(classification_report(encoded_labels_validate, y_pred_validate, target_names=class_names))

