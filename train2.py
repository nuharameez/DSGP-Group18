import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
    class_mode='binary'  # Change to binary classification
)

validation_generator = validation_datagen.flow_from_directory(
    r"C:\Users\multi\Desktop\knee\validate",  # Updated path to validation images
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'  # Change to binary classification
)

# Experiment with deeper architecture and LeakyReLU
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Add dropout for regularization
    Dense(64),
    LeakyReLU(alpha=0.01),
    Dropout(0.5),  # Add dropout for regularization
    Dense(1, activation='sigmoid')
])

# Tune learning rate and batch size
model.compile(optimizer=Adam(learning_rate=0.0001),  # Adjusted learning rate
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Assign higher weights to the osteoarthritis class
class_weights = {0: 1.0, 1: 2.0}  # Adjusted class weights

# Get the number of steps for training and validation
steps_per_epoch = len(train_generator)
validation_steps = len(validation_generator)

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=5,  # Increased training epochs
    validation_data=validation_generator,
    validation_steps=validation_steps,
    class_weight=class_weights
)

model.save('knee_xray_classifier.h5')
