import cv2
import os

import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def load_data_from_folder(data_folder):
    # Lists to store image paths and corresponding labels
    x_paths = []
    y_labels = []

    # Loop through each subdirectory (KL grade)
    for kl_grade in os.listdir(data_folder):
        kl_path = os.path.join(data_folder, kl_grade)

        # Assuming each subdirectory contains image files
        for img_file in os.listdir(kl_path):
            img_path = os.path.join(kl_path, img_file)

            # Read the KL grade from the subdirectory name
            label = int(kl_grade.replace("KL", ""))

            x_paths.append(img_path)
            y_labels.append(label)

    # Load images and labels
    X = []
    y = []

    for img_path, label in zip(x_paths, y_labels):
        # Read and resize the image
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))  # Adjust dimensions as needed

        # Preprocess the image (normalize pixel values to [0, 1])
        img = img / 255.0

        # Append the image and label to the lists
        X.append(img)
        y.append(label)

    return np.array(X), np.array(y)


# Specify the paths for training and testing data folders
train_data_directory = r"C:\Users\MSI\Downloads\IIT STUFF\CM 2603 DS\CW implementation testing\DATASETS\CustomTrain"
test_data_directory = r"C:\Users\MSI\Downloads\IIT STUFF\CM 2603 DS\CW implementation testing\DATASETS\CustomTest"

# Load training data
X_train, y_train = load_data_from_folder(train_data_directory)

# Load testing data
X_test, y_test = load_data_from_folder(test_data_directory)

# Flatten or reshape the images
X_train_flattened = X_train.reshape(X_train.shape[0], -1)
X_test_flattened = X_test.reshape(X_test.shape[0], -1)

# Initialize the decision tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Train the model
clf.fit(X_train_flattened, y_train)

# Make predictions on the test set
y_predict = clf.predict(X_test_flattened)

# Evaluate the model
accuracy = accuracy_score(y_test, y_predict)
print(f"Accuracy: {accuracy:.2f}")

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_predict, zero_division=1))

# Calculate confusion matrix
# cm = confusion_matrix(y_test, y_predict)
#
# # Plot confusion matrix
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
# plt.xlabel("Predicted")
# plt.ylabel("True")
# plt.title("Confusion Matrix")
# plt.show()


# Load and preprocess the input image
input_image_path = r"C:\Users\MSI\Downloads\IIT STUFF\CM 2603 DS\CW implementation testing\DATASETS\train\2\9755634L.png"
input_img = cv2.imread(input_image_path)
input_img = cv2.resize(input_img, (224, 224))
input_img = input_img / 255.0

input_img_flattened = input_img.reshape(1, -1)

prediction = clf.predict(input_img_flattened)

classification_grade = f"KL{prediction[0]}"
print(f"Classification Grade: {classification_grade}")