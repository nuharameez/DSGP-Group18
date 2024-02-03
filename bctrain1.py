import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Define paths
train_path = r"C:\Users\multi\Desktop\All Folders\KneeKaggle\train"
validate_path = r"C:\Users\multi\Desktop\All Folders\KneeKaggle\val"

# Image size and batch size
img_size = (224, 224)
batch_size = 32

# Create data generators
train_datagen = ImageDataGenerator(rescale=1./255)
validate_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True
)

validate_generator = validate_datagen.flow_from_directory(
    validate_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# Load pre-trained model (e.g., MobileNetV2)
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False

# Build the model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validate_generator
)

# Save the model
model.save('knee_model1.h5')


# Evaluate on train data
train_predictions = model.predict(train_generator)
train_predictions = np.round(train_predictions).flatten()
train_labels = train_generator.classes

# Evaluate on validate data
validate_predictions = model.predict(validate_generator)
validate_predictions = np.round(validate_predictions).flatten()
validate_labels = validate_generator.classes

# Confusion Matrix
train_cm = confusion_matrix(train_labels, train_predictions)
validate_cm = confusion_matrix(validate_labels, validate_predictions)

print("Confusion Matrix - Train Data:")
print(train_cm)

print("Confusion Matrix - Validate Data:")
print(validate_cm)

# Classification Report
#train_class_report = classification_report(train_labels, train_predictions, target_names=["normal", "abnormal"])
validate_class_report = classification_report(validate_labels, validate_predictions, target_names=["normal", "abnormal"])


print("Classification Report:")
print(validate_class_report)
