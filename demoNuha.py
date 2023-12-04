import os
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import normalize
from sklearn.model_selection import train_test_split

# Function to load images and labels from a given folder
def load_images_from_folder(folder_path, label):
    dataset = []
    for image_name in os.listdir(folder_path):
        if image_name.endswith('.png'):
            image = cv2.imread(os.path.join(folder_path, image_name))
            image = Image.fromarray(image, 'RGB')
            image = image.resize((SIZE, SIZE))
            dataset.append(np.array(image))
            label.append(label)

# Set the image directory
image_directory = 'C:/Users/AUSU/Desktop/x-rays/'
SIZE = 150

# Initialize lists to store data and labels
dataset = []
label = []

# Load normal images
normal_label = 0
normal_folder_path = os.path.join(image_directory, 'train', 'normal')
load_images_from_folder(normal_folder_path, normal_label)

# Load not normal images
not_normal_label = 1
not_normal_folder_path = os.path.join(image_directory, 'train', 'not normal')
load_images_from_folder(not_normal_folder_path, not_normal_label)

# Convert lists to arrays
dataset = np.array(dataset)
label = np.array(label)

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size=0.20, random_state=0)

# Normalize the data
X_train = normalize(X_train, axis=1)
X_test = normalize(X_test, axis=1)

# Build the model
INPUT_SHAPE = (SIZE, SIZE, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=INPUT_SHAPE))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# ... (continue with the rest of your model architecture)

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Display model summary
print(model.summary())

# Train the model
history = model.fit(X_train,
                    y_train,
                    batch_size=64,
                    verbose=1,
                    epochs=10,
                    validation_data=(X_test, y_test),
                    shuffle=False)

# Save the trained model
model.save('malaria_model_10epochs.h5')

# Corrected index to avoid out-of-bounds error
n = 0  # Select the index of the image to be loaded for testing
img = X_test[n]
plt.imshow(img)
input_img = np.expand_dims(img, axis=0)
image_path = os.path.join(image_directory, 'train', 'normal' if y_test[n] == 0 else 'not normal', os.listdir(os.path.join(image_directory, 'train', 'normal' if y_test[n] == 0 else 'not normal'))[n])
print("Image Path:", image_path)
print("The prediction for this image is: ", model.predict(input_img))
print("The actual label for this image is: ", y_test[n])

# Load the trained model
model = load_model('malaria_model_10epochs.h5')

_, acc = model.evaluate(X_test, y_test)
print("Accuracy = ", (acc * 100.0), "%")
