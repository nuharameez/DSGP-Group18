import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.optimizers import Adam

# Function to load and preprocess images
def load_and_preprocess(folder_path, label):
    data = []
    labels = []

    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = load_img(img_path, target_size=(128, 128))
        img_array = img_to_array(img)
        img_array = cv2.cvtColor(np.array(img_array, dtype=np.uint8), cv2.COLOR_BGR2RGB)
        img_array = cv2.resize(img_array, (128, 128))
        img_array = cv2.GaussianBlur(img_array, (5, 5), 0)

        data.append(img_array)
        labels.append(label)

    return np.array(data), np.array(labels)

# Load and preprocess normal images
normal_data, normal_labels = load_and_preprocess(r"C:\Users\multi\Desktop\KneeKaggle\train\0", 0)

# Load and preprocess osteoarthritis images
osteo_data, osteo_labels = load_and_preprocess(r"C:\Users\multi\Desktop\KneeKaggle\train\4", 1)

# Combine normal and osteoarthritis data
data = np.vstack((normal_data, osteo_data))
labels = np.hstack((normal_labels, osteo_labels))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen.fit(X_train)

# Manually set class weights
class_counts = np.bincount(y_train)
total_samples = len(y_train)
class_weights = total_samples / (len(np.unique(y_train)) * class_counts)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

# Define the model with increased complexity
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))  # Add dropout layer
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Compile the model with adjusted class weights and learning rate
model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with increased epochs
model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=20,
          validation_data=(X_test, y_test), class_weight=class_weight_dict)

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype(int)

print(confusion_matrix(y_test, y_pred_classes))
print(classification_report(y_test, y_pred_classes))

# Save the model
model.save('binary_classification_model_updated.keras')
