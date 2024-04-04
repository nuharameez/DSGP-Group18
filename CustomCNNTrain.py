import os
import cv2
import numpy as np
from keras.src.layers import UpSampling2D
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.applications import VGG16
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import EarlyStopping

data_path = r'C:\Users\MSI\Downloads\IIT STUFF\CM 2603 DS\CW DATASETS\train'
categories = os.listdir(data_path)
labels = [i for i in range(len(categories))]

label_dict = dict(zip(categories, labels))  # empty dictionary

img_size = 256
data = []
label = []

for category in categories:
    folder_path = os.path.join(data_path, category)
    img_names = os.listdir(folder_path)

    for img_name in img_names:
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        try:
            # Convert grayscale image to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(img_rgb, (img_size, img_size))
            data.append(resized)
            label.append(label_dict[category])
        except Exception as e:
            print('Exception:', e)

data = np.array(data) / 255.0
label = np.array(label)
new_label = to_categorical(label, num_classes=4)

# Load pre-trained VGG16 model without the top layer
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))

# Freeze the layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Resize the output of the base model to match the input size expected by the custom CNN
x = UpSampling2D(size=(int(img_size/8), int(img_size/8)))(base_model.output)

# Add a 1x1 Conv2D layer to reduce the number of channels to 3
x = Conv2D(3, (1, 1), activation='relu')(x)

# Define a custom CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu'))
model.add(Dense(4, activation='softmax'))

# Combine VGG16 base and custom model
combined_model = Model(inputs=base_model.input, outputs=model(x))

combined_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

x_train, x_test, y_train, y_test = train_test_split(data, new_label, test_size=0.1)

# Define early stopping
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

history = combined_model.fit(x_train, y_train, epochs=100, validation_split=0.2, callbacks=[early_stopping])

val_loss, val_accuracy = combined_model.evaluate(x_test, y_test, verbose=0)
print("test loss:", val_loss, '%')
print("test accuracy:", val_accuracy, "%")

test_labels = np.argmax(y_test, axis=1)
predictions = combined_model.predict(x_test)
predictions = np.argmax(predictions, axis=-1)

# Print classification report
print(classification_report(test_labels, predictions))

# Print confusion matrix
conf_matrix = confusion_matrix(test_labels, predictions)
print("Confusion Matrix:")
print(conf_matrix)

combined_model.save('Custom_CNN_with_VGG16opti.h5')  # save confusion matrix and classification report
