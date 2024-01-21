# import os
# import cv2
# import numpy as np
# from keras.applications import VGG16
# from keras.applications.vgg16 import preprocess_input
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import GridSearchCV
#
# # Step 1: Load and Display X-ray Images
# def load_images_from_folder(folder):
#     images = []
#     labels = []
#     for subfolder in os.listdir(folder):
#         subfolder_path = os.path.join(folder, subfolder)
#         if os.path.isdir(subfolder_path):
#             label = int(subfolder)  # Assuming folder names are the KL grades
#             for filename in os.listdir(subfolder_path):
#                 img = cv2.imread(os.path.join(subfolder_path, filename), cv2.IMREAD_GRAYSCALE)
#                 if img is not None:
#                     images.append(img)
#                     labels.append(label)
#     return images, labels
#
# train_folder =  r"C:\Users\MSI\Downloads\IIT STUFF\CM 2603 DS\CW implementation testing\DATASETS\CustomTrain"
# validate_folder = r"C:\Users\MSI\Downloads\IIT STUFF\CM 2603 DS\CW implementation testing\DATASETS\CustomVal"
#
# # Load images and labels from train and validate folders
# X_train_images, y_train = load_images_from_folder(train_folder)
# X_validate_images, y_validate = load_images_from_folder(validate_folder)
#
# # Step 2: Image Preprocessing
# def preprocess_images(images):
#     processed_images = [cv2.resize(img, (224, 224))/255.0 for img in images]
#     processed_images = [cv2.convertScaleAbs(img) for img in processed_images]  # Convert to CV_8U
#     processed_images = [cv2.fastNlMeansDenoising(img, None, h=10, templateWindowSize=7, searchWindowSize=21) for img in processed_images]
#     return processed_images
#
# X_train = preprocess_images(X_train_images)
# X_validate = preprocess_images(X_validate_images)
#
# # Step 3: Feature Extraction using a Pre-trained Model
# class VGG16FeatureExtractor:
#     def __init__(self):
#         self.model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
#
#     def transform(self, X):
#         X_preprocessed = [preprocess_input(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)) for img in X]
#         features = [self.model.predict(np.expand_dims(img, axis=0)).flatten() for img in X_preprocessed]
#         return np.array(features)
#
# # Step 4: Train Decision Tree Classifier
# # Create VGG16 feature extractor
# vgg16_extractor = VGG16FeatureExtractor()
#
# # Extract features from X-ray images
# X_train_features = vgg16_extractor.transform(X_train)
# X_validate_features = vgg16_extractor.transform(X_validate)
#
# # Create and train decision tree classifier
# decision_tree = DecisionTreeClassifier(random_state=42)
#
# # Hyperparameter tuning using GridSearchCV
# param_grid = {'max_depth': [None, 10, 20, 30],
#               'min_samples_split': [2, 5, 10],
#               'min_samples_leaf': [1, 2, 4]}
#
# grid_search = GridSearchCV(decision_tree, param_grid, cv=5, scoring='accuracy')
# grid_search.fit(X_train_features, y_train)
#
# best_params = grid_search.best_params_
# print(f'Best Hyperparameters: {best_params}')
#
# # Train the decision tree with the best hyperparameters
# decision_tree = DecisionTreeClassifier(random_state=42, **best_params)
# decision_tree.fit(X_train_features, y_train)
#
# # Make predictions on the validation set
# y_validate_pred = decision_tree.predict(X_validate_features)
#
# # Evaluate accuracy on the validation set
# accuracy = accuracy_score(y_validate, y_validate_pred)
# print(f'Validation Accuracy: {accuracy}')

import os
import numpy as np
from keras.preprocessing import image
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import export_text
from keras.preprocessing.image import ImageDataGenerator

# Define the paths to your dataset
base_dir = r"C:\Users\MSI\Downloads\IIT STUFF\CM 2603 DS\CW implementation testing\DATASETS"
train_dir = os.path.join(base_dir, 'CustomTrain')
validate_dir = os.path.join(base_dir, 'CustomVal')

# Data augmentation using ImageDataGenerator
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Function to load and preprocess images from a directory
def load_and_preprocess(directory):
    image_data = []
    labels = []
    for kl_grade in os.listdir(directory):
        kl_grade_path = os.path.join(directory, kl_grade)
        if os.path.isdir(kl_grade_path):
            for filename in os.listdir(kl_grade_path):
                img_path = os.path.join(kl_grade_path, filename)
                img = load_and_preprocess_image(img_path)
                image_data.append(img)
                labels.append(kl_grade)
    return np.array(image_data), np.array(labels)

# Function to load and preprocess a single image
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = datagen.standardize(img_array)
    return img_array[0]

# Load and preprocess training data
X_train, y_train = load_and_preprocess(train_dir)

# Load and preprocess validation data
X_validate, y_validate = load_and_preprocess(validate_dir)

# Encode class labels
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_validate_encoded = le.transform(y_validate)

# Train a Decision Tree Classifier with hyperparameter tuning
param_grid = {'max_depth': [None, 10, 20, 30],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 2, 4]}

dt_classifier = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(dt_classifier, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train.reshape(X_train.shape[0], -1), y_train_encoded)

# Get the best hyperparameters
best_params = grid_search.best_params_
print(f"Best Hyperparameters: {best_params}")

# Train the model with the best hyperparameters
best_dt_classifier = DecisionTreeClassifier(**best_params, random_state=42)
best_dt_classifier.fit(X_train.reshape(X_train.shape[0], -1), y_train_encoded)

# Make predictions on the validation set
y_pred_validate = best_dt_classifier.predict(X_validate.reshape(X_validate.shape[0], -1))

# Evaluate the model on the validation set
accuracy_validate = accuracy_score(y_validate_encoded, y_pred_validate)
classification_rep_validate = classification_report(y_validate_encoded, y_pred_validate)

print(f"Validation Accuracy: {accuracy_validate}")
print(f"Validation Classification Report:\n{classification_rep_validate}")

# Optional: Print the decision tree rules
tree_rules = export_text(best_dt_classifier, feature_names=list(range(X_train.shape[1])))
print(f"Decision Tree Rules:\n{tree_rules}")

