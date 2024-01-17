#using vgg16 and image data generator

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16

# Function to load and preprocess images
def load_and_preprocess(folder_path, label):
    data = []
    labels = []

    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = load_img(img_path, target_size=(224, 224))  # VGG16 input size
        img_array = img_to_array(img)
        img_array = cv2.cvtColor(np.array(img_array, dtype=np.uint8), cv2.COLOR_BGR2RGB)
        img_array = cv2.resize(img_array, (224, 224))
        img_array = cv2.GaussianBlur(img_array, (5, 5), 0)

        data.append(img_array)
        labels.append(label)

    return np.array(data), np.array(labels)

# Load and preprocess normal images for training
normal_data, normal_labels = load_and_preprocess(r"C:\Users\multi\Desktop\All Folders\KneeKaggle\train\0", 0)

# Load and preprocess osteoarthritis images for training
osteo_data, osteo_labels = load_and_preprocess(r"C:\Users\multi\Desktop\All Folders\KneeKaggle\train\4", 1)

# Combine normal and osteoarthritis training data
data = np.vstack((normal_data, osteo_data))
labels = np.hstack((normal_labels, osteo_labels))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Load and preprocess normal images for validation
val_normal_data, val_normal_labels = load_and_preprocess(r"C:\Users\multi\Desktop\All Folders\KneeKaggle\val\0", 0)

# Load and preprocess osteoarthritis images for validation
val_osteo_data, val_osteo_labels = load_and_preprocess(r"C:\Users\multi\Desktop\All Folders\KneeKaggle\val\4", 1)

# Combine normal and osteoarthritis validation data
val_data = np.vstack((val_normal_data, val_osteo_data))
val_labels = np.hstack((val_normal_labels, val_osteo_labels))

# Data Augmentation for training and validation sets
train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator()

train_datagen.fit(X_train)
val_datagen.fit(val_data)

# Manually set class weights
class_counts = np.bincount(y_train)
total_samples = len(y_train)
class_weights = total_samples / (len(np.unique(y_train)) * class_counts)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

# Load VGG16 model with pre-trained weights (excluding the top dense layers)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Build the model on top of the base_model
model = models.Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

# Compile the model with adjusted class weights and learning rate
model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with increased epochs and validation data
history = model.fit(train_datagen.flow(X_train, y_train, batch_size=32), epochs=20,
                    validation_data=val_datagen.flow(val_data, val_labels, batch_size=32),
                    class_weight=class_weight_dict)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype(int)

print("Test Set Metrics:")
print(confusion_matrix(y_test, y_pred_classes))
print(classification_report(y_test, y_pred_classes))

# Evaluate the model on the validation set
val_pred = model.predict(val_data)
val_pred_classes = (val_pred > 0.5).astype(int)

print("\nValidation Set Metrics:")
print(confusion_matrix(val_labels, val_pred_classes))
print(classification_report(val_labels, val_pred_classes))

# Save the model
model.save('vgg16_transfer_learning_model.keras')
