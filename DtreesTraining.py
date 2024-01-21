import cv2
import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from keras.preprocessing.image import load_img, img_to_array
from joblib import dump


def load_and_preprocess(data_folder):
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
        # Read and preprocess the image
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = cv2.cvtColor(np.array(img_array, dtype=np.uint8), cv2.COLOR_BGR2RGB)
        img_array = cv2.resize(img_array, (224, 224))
        img_array = cv2.GaussianBlur(img_array, (5, 5), 0)
        img_array = img_array / 255.0  # Normalize pixel values to [0, 1]

        # Append the image and label to the lists
        X.append(img_array)
        y.append(label)

    return np.array(X), np.array(y)


# Specify the paths for training and testing data folders
train_data_directory = r"C:\Users\MSI\Downloads\IIT STUFF\CM 2603 DS\CW implementation testing\DATASETS\train"
validate_data_directory = r"C:\Users\MSI\Downloads\IIT STUFF\CM 2603 DS\CW implementation testing\DATASETS\val"

# Load and preprocess training data
X_train, y_train = load_and_preprocess(train_data_directory)

# Load and preprocess validation data
X_val, y_val = load_and_preprocess(validate_data_directory)

# Flatten or reshape the images
X_train_flattened = X_train.reshape(X_train.shape[0], -1)
X_val_flattened = X_val.reshape(X_val.shape[0], -1)

# Initialize the decision tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Train the model on the training set
clf.fit(X_train_flattened, y_train)

# Evaluate the model on the validation set
accuracy = clf.score(X_val_flattened, y_val)
print(f"Validation Accuracy: {accuracy}")

# Save the trained model
model_filename = "decision_tree_model2.joblib"
dump(clf, model_filename)
print(f"Trained model saved as {model_filename}")
