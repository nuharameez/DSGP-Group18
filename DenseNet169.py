import os
import numpy as np
import cv2
from skimage import io
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from keras.applications.densenet import DenseNet169, preprocess_input
from keras.models import Model
from skimage.transform import resize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


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


# Function to load and preprocess images with DenseNet169
def load_images_with_densenet(folder_path, label, model):
    images = []
    labels = []
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith('.png'):
            img_path = os.path.join(folder_path, filename)
            img = io.imread(img_path)
            # img = square(img)
            img = preprocess(img)
            img = resize_crop(img)

            # Preprocess input for DenseNet169
            img = preprocess_input(img.reshape(1, 224, 224, 3))

            # Extract features using DenseNet169
            features = model.predict(img)

            images.append(features.flatten())
            labels.append(label)

    return np.array(images), np.array(labels)


# Load DenseNet169 model
base_model = DenseNet169(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

# Set the path to your dataset
dataset_path = r'C:\Users\MSI\Downloads\IIT STUFF\CM 2603 DS\CW implementation testing\DATASETS'

# Load and preprocess train data
train_data = []
train_labels = []
for kl_grade in range(1, 5):
    kl_folder_path = os.path.join(dataset_path, 'train', str(kl_grade))
    images, labels = load_images_with_densenet(kl_folder_path, kl_grade, model)
    train_data.extend(images)
    train_labels.extend(labels)

# Load and preprocess validate data
validate_data = []
validate_labels = []
for kl_grade in range(1, 5):
    kl_folder_path = os.path.join(dataset_path, 'val', str(kl_grade))
    images, labels = load_images_with_densenet(kl_folder_path, kl_grade, model)
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
model = xgb.XGBClassifier(max_depth=3, subsample=0.7, objective='multi:softmax',
                          num_class=len(np.unique(encoded_labels_train)))

# Train the model
model.fit(X_train, encoded_labels_train)

# Make predictions on the validate set
y_pred_validate = model.predict(X_validate)

# Evaluate the model on validate set
accuracy_validate = accuracy_score(encoded_labels_validate, y_pred_validate)
print(f'Validation Accuracy: {accuracy_validate}')

# Convert class names to strings
class_names = list(map(str, label_encoder.classes_))

# Print classification report
print(classification_report(encoded_labels_validate, y_pred_validate, target_names=class_names))

