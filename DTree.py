import os
import cv2
import numpy as np
from skimage import feature
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img, img_to_array

# Load the pre-trained VGG16 model (excluding the top layer)
vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


# Function for image preprocessing using VGG16 standards
def preprocess_image_vgg16(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = cv2.cvtColor(np.array(img_array, dtype=np.uint8), cv2.COLOR_BGR2RGB)
    img_array = cv2.resize(img_array, (224, 224))
    img_array = cv2.GaussianBlur(img_array, (5, 5), 0)
    img_array = preprocess_input(img_array)  # VGG16 preprocessing
    return img_array


# Function for feature extraction using VGG16 model
def extract_features_vgg16(image):
    features = vgg16_model.predict(np.expand_dims(image, axis=0))
    flattened_features = features.flatten()
    return flattened_features


# Load dataset
data_folder = r"C:\Users\MSI\Downloads\IIT STUFF\CM 2603 DS\CW implementation testing\DATASETS"  # Change this to the actual path
kl_grades = ["1", "2", "3", "4"]

image_paths = []
labels = []

for grade in kl_grades:
    grade_folder = os.path.join(data_folder, "CustomTrain", grade)
    images = os.listdir(grade_folder)

    for image_name in images:
        image_path = os.path.join(grade_folder, image_name)
        image_paths.append(image_path)
        labels.append(int(grade))

# Preprocess images using VGG16 standards and extract features
processed_images = [preprocess_image_vgg16(image_path) for image_path in image_paths]
features_vgg16 = [extract_features_vgg16(image) for image in processed_images]

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(features_vgg16, labels, test_size=0.2, random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {'max_depth': [None, 10, 20, 30],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 2, 4]}

clf = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5)
clf.fit(X_train, y_train)

# Print the best parameters found by GridSearchCV
print("Best Parameters:", clf.best_params_)

# Make predictions on the validation set
val_predictions = clf.predict(X_val)

# Evaluate the model on the validation set
accuracy = accuracy_score(y_val, val_predictions)
print(f"Validation Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(y_val, val_predictions))
