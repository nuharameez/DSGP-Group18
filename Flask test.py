# Import necessary libraries
from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load your dataset
df = pd.read_csv('Treatement dataset1.csv')

# Convert categorical variables to one-hot encoding
df_encoded = pd.get_dummies(df, columns=['Treatment Recommendation'], prefix='', prefix_sep='')

# Assuming 'Level' is the target variable and the rest are features
X = df_encoded.drop('Level', axis=1)
y = df_encoded['Level']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest model with class_weight parameter
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

# Train the model
rf_model.fit(X_train, y_train)

# Save the trained model to a file
model_filename = 'random_forest_model.joblib'
joblib.dump(rf_model, model_filename)

# Create a Flask application
app = Flask(__name__)

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for handling form submission
@app.route('/recommendations', methods=['POST'])
def recommendations():
    # Get user-selected grade from the form
    user_grade = request.form['grade']

    # Extract treatments for the specific grade
    grade_treatments = df[df['Level'] == f'Grade {user_grade}']['Treatment Recommendation'].tolist()

    # Provide a small description for each grade
    grade_descriptions = {
        '0': "Grade 0 - No evidence of osteoarthritis.",
        '1': "Grade 1 - Doubtful narrowing of joint space and possible osteophytic lipping.",
        '2': "Grade 2 - Definite osteophytes and possible narrowing of joint space.",
        '3': "Grade 3 - Moderate multiple osteophytes, definite narrowing of joints space, some sclerosis, and possible deformity of bone ends.",
        '4': "Grade 4 - Large osteophytes, marked narrowing of joint space, severe sclerosis, and definite deformity of bone ends."
    }

    # Render the template with grade description and treatment recommendations
    return render_template('recommendations.html', grade_description=grade_descriptions[user_grade], treatments=grade_treatments)

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)

"""from flask import Flask, render_template, request
import joblib
import pandas as pd

# Load the trained model
model_filename = 'random_forest_model.joblib'
rf_model = joblib.load(model_filename)

# Load your dataset
df = pd.read_csv('Treatement dataset1.csv')

app = Flask(__name__)

# Function to recommend treatments for a specific grade
def recommend_treatments_for_grade(grade):
    # Extract treatments for the specific grade
    grade_treatments = df[df['Level'] == f'Grade {grade}']['Treatment Recommendation'].tolist()

    if grade_treatments:
        return grade_treatments
    else:
        return ["No treatment recommendations available for Grade {grade}."]

# Function to print description for a specific grade
def print_grade_description(grade):
    # Provide a small description for each grade
    grade_descriptions = {
        '0': "Grade 0 - No evidence of osteoarthritis.",
        '1': "Grade 1 - Doubtful narrowing of joint space and possible osteophytic lipping.",
        '2': "Grade 2 - Definite osteophytes and possible narrowing of joint space.",
        '3': "Grade 3 - Moderate multiple osteophytes, definite narrowing of joints space, some sclerosis, and possible deformity of bone ends.",
        '4': "Grade 4 - Large osteophytes, marked narrowing of joint space, severe sclerosis, and definite deformity of bone ends."
    }

    return grade_descriptions.get(grade, "Invalid grade. Please enter a valid grade.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    user_grade = request.form['grade']
    grade_description = print_grade_description(user_grade)
    treatments = recommend_treatments_for_grade(user_grade)
    return render_template('recommendations.html', grade_description=grade_description, treatments=treatments)

if __name__ == '__main__':
    app.run(debug=True)"""
