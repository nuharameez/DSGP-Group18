import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from imblearn.over_sampling import RandomOverSampler

# Set the seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Function to load and preprocess images
def load_and_preprocess_images(directory):
    images = []
    labels = []
    for label in os.listdir(directory):
        label_path = os.path.join(directory, label)
        for image_file in os.listdir(label_path):
            image_path = os.path.join(label_path, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.GaussianBlur(image, (5, 5), 0)
            image = cv2.Canny(image, 50, 150)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Convert to 3 channels for VGG16
            image = cv2.resize(image, (224, 224))  # Resize for VGG16 input size
            image = image.astype('float32') / 255.0  # Normalize pixel values to between 0 and 1
            images.append(image)
            labels.append(int(label))  # Assuming binary classification with 0 and 1 labels
    return np.array(images), np.array(labels)

# Load and preprocess training data
train_dir = r"C:\Users\multi\Desktop\All Folders\KneeKaggle\train"
X_train, y_train = load_and_preprocess_images(train_dir)

# Apply oversampling to address class imbalance
oversampler = RandomOverSampler(sampling_strategy='auto')
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train.reshape(-1, 224 * 224 * 3), y_train)
X_train_resampled = X_train_resampled.reshape(-1, 224, 224, 3)

# Load and preprocess validation data
val_dir = r"C:\Users\multi\Desktop\All Folders\KneeKaggle\val"
X_val, y_val = load_and_preprocess_images(val_dir)

# Create VGG16 base model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the convolutional layers
for layer in base_model.layers:
    layer.trainable = False

# Create the model
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Create data generators with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow(X_train_resampled, y_train_resampled, batch_size=32, shuffle=True)
val_generator = val_datagen.flow(X_val, y_val, batch_size=32, shuffle=False)

# Train the model
model.fit(train_generator, epochs=10, validation_data=val_generator)

# Save the model
model.save('binary_classification_model_with_augmentation.keras')