import tensorflow as tf
from keras.src.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import scipy

# Specify image dimensions and batch size
img_width, img_height = 150, 150
batch_size = 32

# Enhanced data augmentation with more variations
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=40,  # Add image rotations
    width_shift_range=0.2,
    height_shift_range=0.2  # Add random shifts
)


validation_datagen = ImageDataGenerator(rescale=1./255)

# Load images from directories
train_generator = train_datagen.flow_from_directory(
    r"C:\Users\multi\Desktop\knee\train",  # Updated path to training images
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    r"C:\Users\multi\Desktop\knee\validate",  # Updated path to validation images
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

# Experiment with deeper architecture and LeakyReLU
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),  # Additional layer
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),  # Increased neurons
    Dense(64, activation='leaky_relu'),  # Experiment with LeakyReLU
    Dense(1, activation='sigmoid')
])

# Tune learning rate and batch size
model.compile(optimizer=Adam(learning_rate=0.0005),  # Lower learning rate
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Smaller batch size for potential better convergence
batch_size = 16


epochs = 20
steps_per_epoch = len(train_generator)
validation_steps = len(validation_generator)

# Assign higher weights to the normal class
class_weights = {0: 1.0, 1: 2.0}  # Assuming 0 for normal, 1 for osteoarthritis

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=40,  # Increased training epochs
    validation_data=validation_generator,
    validation_steps=validation_steps,
    class_weight=class_weights
)


model.save('knee_xray_classifier.h5')
# Load the model
new_model = tf.keras.models.load_model('knee_xray_classifier.h5')


