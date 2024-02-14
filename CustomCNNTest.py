import os
import cv2
import numpy as np
from keras.models import load_model
from keras.src.utils import to_categorical
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt

# Load the saved model
saved_model_path = 'Custom_CNN_with_VGG16.h5'
saved_model = load_model(saved_model_path)

# Load testing data
test_data_path = r'C:\Users\MSI\Downloads\IIT STUFF\CM 2603 DS\CW DATASETS\auto_test'  # Change this to your testing data path
categories = os.listdir(test_data_path)
labels = [i for i in range(len(categories))]

label_dict = dict(zip(categories, labels))  # empty dictionary

img_size = 256
test_data = []
test_label = []

for category in categories:
    folder_path = os.path.join(test_data_path, category)
    img_names = os.listdir(folder_path)

    for img_name in img_names:
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        try:
            # Convert grayscale image to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(img_rgb, (img_size, img_size))
            test_data.append(resized)
            test_label.append(label_dict[category])
        except Exception as e:
            print('Exception:', e)

test_data = np.array(test_data) / 255.0
test_label = np.array(test_label)
# Encode the target labels
test_label = to_categorical(test_label, num_classes=len(categories))

# Evaluate the model
test_loss, test_accuracy = saved_model.evaluate(test_data, test_label, verbose=0)
print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)

# Generate predictions
predictions = saved_model.predict(test_data)
predicted_labels = np.argmax(predictions, axis=1)

# Create confusion matrix
conf_matrix = confusion_matrix(np.argmax(test_label, axis=1), predicted_labels)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plot_confusion_matrix(conf_matrix, figsize=(12, 8), class_names=categories)
plt.show()