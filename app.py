import os
from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
from keras.models import load_model

app = Flask(__name__)

# Load the saved model
model_path = 'Custom_CNN_with_VGG16.h5'
model = load_model(model_path)

# Define the categories
categories = ['category1', 'category2', 'category3', 'category4']

# Function to preprocess uploaded image
def preprocess_image(image_path):
    img_size = 256
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (img_size, img_size))
    return np.expand_dims(resized / 255.0, axis=0)

# Function to get the grade for the image
def get_grade(image_path):
    preprocessed_img = preprocess_image(image_path)
    prediction = model.predict(preprocessed_img)
    category_index = np.argmax(prediction)
    return categories[category_index]

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)
            grade = get_grade(file_path)
            return render_template('result.html', grade=grade, image_file=file_path)
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
