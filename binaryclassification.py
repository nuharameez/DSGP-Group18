import tensorflow as tf
from keras.src.saving.saving_api import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
import numpy as np

# Define data paths
train_data_dir = './x-rays/train'
validation_data_dir = './x-rays/validation'
import os
print("Current Working Directory:", os.getcwd())


# Set parameters
input_size = (150, 150)
batch_size = 32
epochs = 10

# Data augmentation for the training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Rescaling for the validation set
validation_datagen = ImageDataGenerator(rescale=1./255)

# Generate batches of augmented data for training and validation
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=input_size,
    batch_size=batch_size,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=input_size,
    batch_size=batch_size,
    class_mode='binary'
)

# Build the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)


# Save the model
model.save('knee_bone_model.h5')

# Test the model on the validation set
# Load the trained model
model = load_model('knee_bone_model.h5')

# Hardcoded path to the image you want to test
image_path = 'C:/Users/AUSU/Desktop/Year 02/DSGP/DSGP-Group18/x-rays/test/2.png'

# Load and preprocess the image
img = image.load_img(image_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Rescale the pixel values to the range [0, 1], same as during training

# Make predictions
predictions = model.predict(img_array)

# Display the prediction result
if predictions[0, 0] > 0.5:
    print("The model predicts: Not Normal")
else:
    print("The model predicts: Normal")


