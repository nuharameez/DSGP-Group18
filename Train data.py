"""# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from imblearn.over_sampling import RandomOverSampler

# Load your dataset
df = pd.read_csv('Treatement dataset1.csv')

# Convert categorical variables to one-hot encoding
df_encoded = pd.get_dummies(df, columns=['Treatment Recommendation'], prefix='', prefix_sep='')

# Assuming 'Level' is the target variable and the rest are features
X = df_encoded.drop('Level', axis=1)
y = df_encoded['Level']

# Apply RandomOverSampler to balance the data
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# Split the resampled data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Initialize the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training set
rf_model.fit(X_train, y_train)

# Calculate accuracy on the training set
y_train_pred = rf_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy:.2f}")

# Validation
y_valid_pred = rf_model.predict(X_valid)
valid_accuracy = accuracy_score(y_valid, y_valid_pred)
print(f"Validation Accuracy: {valid_accuracy:.2f}")

# Make predictions on the test set
y_test_pred = rf_model.predict(X_test)

# Calculate accuracy on the testing set
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Testing Accuracy: {test_accuracy:.2f}")

# Print classification report for testing set
print("Classification Report (Testing Set):\n")
print(classification_report(y_test, y_test_pred))

# Save the trained model to a file
model_filename = 'random_forest_model_balanced.joblib'
joblib.dump(rf_model, model_filename)
print(f"Trained Random Forest model saved as {model_filename}")

# Function to recommend treatments for a specific grade using the trained model
def recommend_treatments_for_grade(grade, model, df_encoded):
    # Convert grade to string with 'Grade'
    grade = f'Grade {grade}'

    # Extract features for the specific grade
    features = df_encoded[df_encoded['Level'] == grade]

    if not features.empty:
        # Drop the 'Level' column as it's not needed for prediction
        features = features.drop('Level', axis=1)

        # Make predictions using the trained model
        treatments = model.predict(features)

        # Print treatment recommendations for the specific grade one by one
        print(f"\nTreatment Recommendations for {grade}:\n")
        for i, treatment in enumerate(treatments, start=1):
            print(f"Treatment {i}: {treatment}")
    else:
        print(f"\nNo treatment recommendations available for {grade}.")

def print_grade_description(grade):
    # Provide a small description for each grade
    grade_descriptions = {
        '0': "Grade 0 - No evidence of osteoarthritis.",
        '1': "Grade 1 - Doubtful narrowing of joint space and possible osteophytic lipping.",
        '2': "Grade 2 - Definite osteophytes and possible narrowing of joint space.",
        '3': "Grade 3 - Moderate multiple osteophytes, definite narrowing of joints space, some sclerosis, and possible deformity of bone ends.",
        '4': "Grade 4 - Large osteophytes, marked narrowing of joint space, severe sclerosis, and definite deformity of bone ends."
    }

    if grade in grade_descriptions:
        print(grade_descriptions[grade])
    else:
        print("Invalid grade. Please enter a valid grade.")

# Load the trained Random Forest model
loaded_model = joblib.load('random_forest_model_balanced.joblib')

# User input for grade
user_grade = input("Enter the grade for treatment recommendations (e.g., 0, 1, 2, ...): ")

# Print grade description
print_grade_description(user_grade)

# Recommend treatments for the user-specified grade using the trained model
recommend_treatments_for_grade(user_grade, loaded_model, df_encoded)"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
df = pd.read_csv('Treatement dataset1.csv')

# Split the data into features (X) and target variable (y)
X = df.drop('Severity', axis=1)  # Features
y = df['Severity']  # Target variable

# Apply one-hot encoding to categorical variables
# Here, we're assuming Treatment is the only categorical variable
ct = ColumnTransformer([('encoder', OneHotEncoder(), ['Treatment'])], remainder='passthrough')
X_encoded = ct.fit_transform(X)

# Split the encoded data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Initialize the random forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save the trained model
joblib.dump(clf, 'rf1.pkl')

import pandas as pd
import joblib

# Function to get treatments for a given severity level
def get_treatments(severity_level, model):
    treatments = df[df['Severity'] == severity_level]['Treatment']
    return treatments.tolist()

# Load the trained model
model = joblib.load('rf1.pkl')

# Test code
while True:
    severity_input = input("Enter severity level (0 to 4) or 'exit' to quit: ")
    if severity_input.lower() == 'exit':
        print("Exiting...")
        break
    try:
        severity_level = int(severity_input)
        if severity_level < 0 or severity_level > 4:
            print("Invalid severity level. Please enter a number between 0 and 4.")
            continue
        severity_treatments = get_treatments(severity_level, model)
        if severity_treatments:
            print(f"Treatments for severity level {severity_level}:")
            for treatment in severity_treatments:
                print("-", treatment)
        else:
            print(f"No treatments found for severity level {severity_level}.")
    except ValueError:
        print("Invalid input. Please enter a valid severity level (0 to 4) or 'exit' to quit.")