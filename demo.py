# import os
# from glob import iglob
# import skimage.io as io
# from skimage.util import img_as_float
# from skimage import exposure
# import skimage.filters as filters
# from skimage.util import img_as_ubyte
# from skimage.transform import resize
# from skimage.morphology import reconstruction
# from scipy.ndimage import gaussian_filter
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import LabelEncoder
#
# # Function for preprocessing an image
# def preprocess_image(img_path):
#     img = io.imread(img_path)
#     img = square(img)
#     img = contrast(img)
#     img = threshold(img)
#     img = resize_crop(img)
#     return img
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
# # Function for enhancing the contrast
# def contrast(img):
#     img = img_as_ubyte(img / 255)
#     img = img_as_float(img)
#     img = gaussian_filter(img, 1)
#     h = 0.8
#     seed = img - h
#     mask = img
#     dilated = reconstruction(seed, mask, method='dilation')
#     img_dil_adapteq = exposure.equalize_adapthist(img - dilated, clip_limit=0.03)
#     return img_dil_adapteq
#
# # Function for thresholding
# def threshold(img):
#     img_threshold = filters.threshold_li(img)
#     img_new = np.ones(img.shape[:2], dtype="float")
#     img_new[(img < img_threshold) | (img > 250)] = 0
#     return img_new
#
# # Function for resizing and cropping
# def resize_crop(img):
#     img = resize(img, (128, 128))
#     img = img[:, 14:115]
#     return img
#
# # Read image paths and their corresponding KL grades
# image_paths_with_kl = []
# for folder in ["CustomTrain", "CustomVal"]:
#     for kl_grade in os.listdir(f"C:/Users/MSI/Downloads/IIT STUFF/CM 2603 DS/CW implementation testing/DATASETS/{folder}"):
#         folder_path = f"C:/Users/MSI/Downloads/IIT STUFF/CM 2603 DS/CW implementation testing/DATASETS/{folder}/{kl_grade}"
#         kl_label = f"KL Grade {kl_grade}"
#         image_paths = [img_path for img_path in iglob(os.path.join(folder_path, "*.png"))]
#         image_paths_with_kl.extend([(img_path, kl_label) for img_path in image_paths])
#
# # Preprocess images and extract features
# X = [preprocess_image(img_path) for img_path, _ in image_paths_with_kl]
# X = np.array(X).reshape(len(X), -1)  # Flatten images
# y = [kl_label for _, kl_label in image_paths_with_kl]
#
# # Convert labels to numeric format
# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)
#
# # Split the data into training and validation sets
# X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
#
# # Create a decision tree classifier
# decision_tree = DecisionTreeClassifier(random_state=42)
#
# # Train the decision tree classifier
# decision_tree.fit(X_train, y_train)
#
# # Make predictions on the validation set
# y_pred = decision_tree.predict(X_val)
#
# # Evaluate the accuracy of the model
# accuracy = accuracy_score(y_val, y_pred)
# print(f"Accuracy: {accuracy}")
#
# # Optionally, you can visualize the decision tree using graphviz
# # from sklearn.tree import export_text
# # tree_rules = export_text(decision_tree, feature_names=list(range(X.shape[1])))
# # print(tree_rules)

from skimage import exposure
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
from sklearn.model_selection import train_test_split
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

# Create an XGBoost model
model = xgb.XGBClassifier(max_depth=6, subsample=0.8, objective='multi:softmax', num_class=len(np.unique(encoded_labels_train)))

# Train the model
model.fit(X_train, encoded_labels_train)

# Save the trained model to a file
# model_filename = 'xgboost_model2.joblib'
# joblib.dump(model, model_filename)
# print(f"Model saved as '{model_filename}'")
#
# label_encoder_filename = 'label_encoder2.joblib'
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
