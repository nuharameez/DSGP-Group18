import cv2
import os

import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Assuming your data is organized in directories by KL grade
data_directory = r"C:\Users\MSI\Downloads\IIT STUFF\CM 2603 DS\CW implementation testing\DATASETS\CustomTrain"

# Lists to store image paths and corresponding labels
X_paths = []
y_labels = []

# Loop through each subdirectory (KL grade)
for kl_grade in os.listdir(data_directory):
    kl_path = os.path.join(data_directory, kl_grade)

    # Assuming each subdirectory contains image files
    for img_file in os.listdir(kl_path):
        img_path = os.path.join(kl_path, img_file)

        # Read the KL grade from the subdirectory name (assuming it's a number)
        label = int(kl_grade.replace("KL", ""))

        X_paths.append(img_path)
        y_labels.append(label)

# Load images and labels
X = []
y = []

for img_path, label in zip(X_paths, y_labels):
    # Read and resize the image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))  # Adjust dimensions as needed

    # Preprocess the image (normalize pixel values to [0, 1])
    img = img / 255.0

    # Append the image and label to the lists
    X.append(img)
    y.append(label)

# Convert lists to NumPy arrays
X = np.array(X)
y = np.array(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Flatten or reshape the images
X_train_flattened = X_train.reshape(X_train.shape[0], -1)
X_test_flattened = X_test.reshape(X_test.shape[0], -1)

# Initialize the decision tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Train the model
clf.fit(X_train_flattened, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test_flattened)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=1))