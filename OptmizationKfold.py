import os
import cv2
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.src.callbacks import EarlyStopping
from keras.src.layers import BatchNormalization
from keras.src.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from keras.layers import Dense, Dropout
from keras.layers import Input
from keras import models
import joblib

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# Function to load and preprocess images
def load_and_preprocess_data(folder_path):
    images = []
    labels = []

    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)
        images.extend(
            [cv2.resize(cv2.imread(os.path.join(label_path, file_name), cv2.IMREAD_COLOR), (224, 224)) for file_name in
             os.listdir(label_path)])
        labels.extend([label] * len(os.listdir(label_path)))

        for file_name in os.listdir(label_path):
            img_path = os.path.join(label_path, file_name)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (224, 224))
            images.append(img)
            labels.append(label)

    return np.array(images), np.array(labels)


# Load and preprocess data
data_path = r"C:\Users\MSI\Downloads\IIT STUFF\CM 2603 DS\CW implementation testing\DATASETS\train"
X, y = load_and_preprocess_data(data_path)

# Convert labels to numerical values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Preprocess images for VGG16
X = preprocess_input(X)

# Load pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Extract features using VGG16
features = base_model.predict(X)
features_flat = features.reshape(features.shape[0], -1)

# Define K-fold cross-validation
kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

# Initialize variables to store performance metrics
accuracy_scores = []

# Initialize variables to store classification reports
classification_reports = []

# Build the model outside the loop
input_layer = Input(shape=(features_flat.shape[1],))
x = Dense(256, activation='relu')(input_layer)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
output_layer = Dense(4, activation='softmax')(x)

model = models.Model(inputs=input_layer, outputs=output_layer)

# Compile the model outside the loop
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Iterate over K folds
for fold, (train_index, val_index) in enumerate(kfold.split(features_flat, y)):
    X_train, X_val = features_flat[train_index], features_flat[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # Convert labels to one-hot encoded format
    y_train_one_hot = to_categorical(y_train, num_classes=4)
    y_val_one_hot = to_categorical(y_val, num_classes=4)

    # Add Early Stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Train the model
    model.fit(X_train, y_train_one_hot, epochs=10, batch_size=32, validation_data=(X_val, y_val_one_hot),
              callbacks=[early_stopping])

    # Evaluate the model on the validation set
    y_pred = np.argmax(model.predict(X_val), axis=1)
    accuracy = accuracy_score(y_val, y_pred)
    accuracy_scores.append(accuracy)

    # Calculate classification report
    class_report = classification_report(y_val, y_pred, target_names=label_encoder.classes_)
    classification_reports.append(class_report)

# Print average accuracy across all folds
print("Average Validation Accuracy:", np.mean(accuracy_scores))

# Print classification reports for each fold
for i, report in enumerate(classification_reports):
    print(f"\nClassification Report - Fold {i+1}:\n{report}")

# Save the trained model
# model.save('VGG16KFold3.h5')
#
# # Save the label encoder
# joblib.dump(label_encoder, 'label_encoder_VGG16KFold3.joblib')


# Function to load and preprocess test images

# Load and preprocess test data
test_data_path = r"C:\Users\MSI\Downloads\IIT STUFF\CM 2603 DS\CW implementation testing\DATASETS\CustomTest"
X_test, y_test = load_and_preprocess_data(test_data_path)

# Convert labels to numerical values using the same label encoder
y_test_encoded = label_encoder.transform(y_test)

# Preprocess images for VGG16
X_test = preprocess_input(X_test)

# Extract features using the pre-trained VGG16 model
features_test = base_model.predict(X_test)
features_flat_test = features_test.reshape(features_test.shape[0], -1)

# Convert labels to one-hot encoded format
y_test_one_hot = to_categorical(y_test_encoded, num_classes=4)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(features_flat_test, y_test_one_hot)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
print(f"Test Loss: {test_loss}")

# Make predictions on the test set
y_pred_test = np.argmax(model.predict(features_flat_test), axis=1)

# Calculate and print the classification report for the test set
test_class_report = classification_report(y_test_encoded, y_pred_test, target_names=label_encoder.classes_)
print("\nClassification Report - Test Set:\n", test_class_report)
