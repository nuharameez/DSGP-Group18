import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Specify image dimensions and batch size
img_width, img_height = 150, 150
batch_size = 16  # Changed batch size

def contrast_adjust(x):
    return tf.image.adjust_contrast(x, 0.8)  # Adjust contrast by 20%

def apply_contrast_adjust(params):
    x = params['x']
    return contrast_adjust(x) * params['contrast_factor']

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    brightness_range=[0.8, 1.2],
    channel_shift_range=20  # Channel shift for contrast adjustment
)

train_datagen.apply_transform = apply_contrast_adjust

# Load images from directories
train_generator = train_datagen.flow_from_directory(
    r"C:\Users\multi\Desktop\knee\train",
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

# Data generator for validation data (no or minimal augmentation)
validation_datagen = ImageDataGenerator(rescale=1./255)  # Rescaling only

# Now you can use validation_datagen to create the validation generator
validation_generator = validation_datagen.flow_from_directory(
    r"C:\Users\multi\Desktop\knee\validate",
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

# Deeper model architecture with LeakyReLU
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),  # Additional layer
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),  # Increased neurons
    Dense(64),
    LeakyReLU(alpha=0.01),  # Add LeakyReLU layer
    Dense(1, activation='sigmoid')
])

# Tuned learning parameters
model.compile(optimizer=Adam(learning_rate=0.0001),  # Lower learning rate
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Class weights to address imbalance
class_weights = {0: 1.0, 1: 6.93}

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=60,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    class_weight=class_weights
)

# Save the model
model.save('knee_xray_classifier2.keras')
