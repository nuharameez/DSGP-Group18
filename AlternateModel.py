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
