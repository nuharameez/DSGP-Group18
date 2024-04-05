import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np

# Define the input shape of your images
input_shape = (224, 224, 3)  # Adjust the dimensions based on your images

# Define the CNN model
model = models.Sequential()

# Convolutional layers
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Flatten the output to feed into dense layers
model.add(layers.Flatten())

# Dense layers
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))  # Optional dropout layer for regularization
model.add(layers.Dense(1, activation='sigmoid'))  # Output layer for binary classification

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()

# Set up data generators to load and preprocess images
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range = 20,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    fill_mode = 'nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'Bone_Xrays/Train',  # Path to your train folder
    target_size=input_shape[:2],
    batch_size=32,
    class_mode='binary',  # Binary classification
    classes=['other bones', 'knee bone'],  # Class labels
    shuffle=True  # Shuffle the data
)

validation_generator = validation_datagen.flow_from_directory(
    'Bone_Xrays/Validate',  # Path to your validate folder
    target_size=input_shape[:2],
    batch_size=32,
    class_mode='binary',
    classes=['other bones', 'knee bone']
)

# Train the model
history = model.fit(
    train_generator,
    epochs=10,  # Adjust the number of epochs based on your needs
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)

# Evaluate the model on the validation set
val_loss, val_acc = model.evaluate(validation_generator, verbose=2)
print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

# Save the model to a file
model.save('knee_bone_identifier.h5')


"""""
# Load the saved model
loaded_model = load_model('knee_bone_identifier.h5')

# Test the loaded model on a new image
img_path = 'Bone_Xrays/Test/00a2145de1886cb9eb88869c85d74080.png'  # Replace with the path to your test image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Rescale to match the training data

# Make predictions using the loaded model
predictions = loaded_model.predict(img_array)

# Display the result
if predictions[0][0] > 0.5:
    print("Predicted: Knee Bone")
else:
    print("Predicted: Non-Knee Bone")
"""""