
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt.style.use('classic')
#############################################################
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
# from keras import backend as K
####################################################
import os
import cv2
from PIL import Image
import numpy as np

image_directory = 'C:/Users/AUSU/Desktop/x-rays/train/'
SIZE = 150
dataset = []
label = []

normal_images = os.listdir(image_directory + 'normal/')
for i, image_name in enumerate(normal_images):
    if image_name.split('.')[1] == 'png':
        image = cv2.imread(os.path.join(image_directory, 'normal', image_name))
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        dataset.append(np.array(image))
        label.append(0)

notnormal_images = os.listdir(image_directory + 'not normal/')
for i, image_name in enumerate(notnormal_images):
    if image_name.split('.')[1] == 'png':
        image = cv2.imread(os.path.join(image_directory, 'not normal', image_name))
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        dataset.append(np.array(image))
        label.append(1)

dataset = np.array(dataset)
label = np.array(label)


from sklearn.model_selection import train_test_split

# from keras.utils import to_categorical

X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size=0.20, random_state=0)

# Without scaling (normalize) the training may not converge.
# Normalization is a rescaling of the data from the original range
# so that all values are within the range of 0 and 1.
from keras.utils import normalize

X_train = normalize(X_train, axis=1)
X_test = normalize(X_test, axis=1)

# Do not do one-hot encoding as it generates a shape of (num, 2)
# But the network expects an input of (num, 1) for the last layer for binary classification
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

##############################################

###2 conv and pool layers. with some normalization and drops in between.

INPUT_SHAPE = (SIZE, SIZE, 3)  # change to (SIZE, SIZE, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=INPUT_SHAPE))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('sigmoid'))
# Do not use softmax for binary classification
# Softmax is useful for mutually exclusive classes, either cat or dog but not both.
# Also, softmax outputs all add to 1. So good for multi class problems where each
# class is given a probability and all add to 1. Highest one wins.

# Sigmoid outputs probability. Can be used for non-mutually exclusive problems.
# But, also good for binary mutually exclusive (cat or not cat).

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',  # also try adam
              metrics=['accuracy'])

print(model.summary())
###############################################################

##########################################################

history = model.fit(X_train,
                    y_train,
                    batch_size=64,
                    verbose=1,
                    epochs=10,
                    validation_data=(X_test, y_test),
                    shuffle=False
                    )

model.save('malaria_model_10epochs.h5')



#########################################################################################
# Test the model on one image (for 300 epochs)
# img 23 is parasitized - correctly predicts near 0 probability
# Img 22, parasitized, correctly lables (low value) but relatively high value.
# img 24 is uninfected, correctly predicts as uninfected
# img 26 is parasitized but incorrectly gives high value for prediction, uninfected.

# Iterate over all images in the testing set
for n in range(len(X_test)):
    img = X_test[n]
    plt.imshow(img)

    # Expand dims so the input is (num images, x, y, c)
    input_img = np.expand_dims(img, axis=0)

    print("Image Path:", image_directory + 'not normal/' + os.listdir(image_directory + 'not normal/')[n])
    print("The prediction for this image is: ", model.predict(input_img))
    print("The actual label for this image is: ", y_test[n])

# Evaluate the model on all test data for accuracy
_, acc = model.evaluate(X_test, y_test)
print("Accuracy = ", (acc * 100.0), "%")
