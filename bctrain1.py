import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class KneeNormalityChecker:
    def __init__(self, train_path, validate_path, test_path, img_size=(224, 224), batch_size=36):
        self.train_path = train_path
        self.validate_path = validate_path
        self.test_path = test_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.train_generator = None
        self.validate_generator = None
        self.test_generator = None
        self.model = None

    def preprocess_image(self):
        train_datagen = ImageDataGenerator(rescale=1./255)
        validate_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)

        self.train_generator = train_datagen.flow_from_directory(
            self.train_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=True
        )

        self.validate_generator = validate_datagen.flow_from_directory(
            self.validate_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False
        )

        self.test_generator = test_datagen.flow_from_directory(
            self.test_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False
        )

    def build_model(self):
        base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                                       include_top=False,
                                                       weights='imagenet')
        base_model.trainable = False

        self.model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])

        self.model.compile(optimizer='adam',
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

    def plot_confusion_matrix(self, cm, labels):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.show()
    def train(self, epochs=50):
        history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.validate_generator
        )

    def validate(self):
        train_predictions = self.model.predict(self.train_generator)
        train_predictions = np.round(train_predictions).flatten()
        train_labels = self.train_generator.classes

        validate_predictions = self.model.predict(self.validate_generator)
        validate_predictions = np.round(validate_predictions).flatten()
        validate_labels = self.validate_generator.classes

        train_cm = confusion_matrix(train_labels, train_predictions)
        validate_cm = confusion_matrix(validate_labels, validate_predictions)

        # Visualize confusion matrix for validation data
        self.plot_confusion_matrix(validate_cm, labels=["normal", "abnormal"])
        print("Confusion Matrix - Train Data:")
        print(train_cm)

        print("Confusion Matrix - Validate Data:")
        print(validate_cm)

        validate_class_report = classification_report(validate_labels, validate_predictions, target_names=["normal", "abnormal"])

        print("Classification Report:")
        print(validate_class_report)

        # Train accuracy
        train_loss, train_accuracy = self.model.evaluate(self.train_generator, verbose=0)
        print("Train Accuracy:", train_accuracy)

        # Validation accuracy
        validate_loss, validate_accuracy = self.model.evaluate(self.validate_generator, verbose=0)
        print("Validation Accuracy:", validate_accuracy)

    def test(self):
        test_predictions = self.model.predict(self.test_generator)
        test_predictions = np.round(test_predictions).flatten()
        test_labels = self.test_generator.classes

        test_cm = confusion_matrix(test_labels, test_predictions)

        # Visualize confusion matrix for test data
        self.plot_confusion_matrix(test_cm, labels=["normal", "abnormal"])

        print("Confusion Matrix - Test Data:")
        print(test_cm)

        test_class_report = classification_report(test_labels, test_predictions, target_names=["normal", "abnormal"])

        print("Classification Report:")
        print(test_class_report)

        # Test accuracy
        test_loss, test_accuracy = self.model.evaluate(self.test_generator, verbose=0)
        print("Test Accuracy:", test_accuracy)

    def save_model(self, filepath):
        self.model.save(filepath)

# Define paths
train_path = r"C:\Users\multi\Desktop\All Folders\KneeKaggle\train"
validate_path = r"C:\Users\multi\Desktop\All Folders\KneeKaggle\val"
test_path = r"C:\Users\multi\Desktop\All Folders\KneeKaggle\test"

# Initialize model
knee_model = KneeNormalityChecker(train_path, validate_path, test_path)

# Preprocess images
knee_model.preprocess_image()

# Build model
knee_model.build_model()

# Train model
knee_model.train()

# Validate model
knee_model.validate()

# Test model
knee_model.test()
# Save model
knee_model.save_model('knee_normality_checker1.h5')
