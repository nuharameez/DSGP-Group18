"""# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib  # Import joblib for saving the model
from sklearn.metrics import confusion_matrix

# Load your dataset
df = pd.read_csv('Treatement dataset1.csv')

# Convert categorical variables to one-hot encoding
df_encoded = pd.get_dummies(df, columns=['Treatment Recommendation'], prefix='', prefix_sep='')

# Assuming 'Level' is the target variable and the rest are features
X = df_encoded.drop('Level', axis=1)
y = df_encoded['Level']

# Split the data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Initialize the Random Forest model with class_weight parameter
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

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

# Save the trained model to a file
model_filename = 'random_forest_model.joblib'
joblib.dump(rf_model, model_filename)
print(f"Trained Random Forest model saved as {model_filename}")

# Classification Report for testing set
classification_report_test = classification_report(y_test, y_test_pred, zero_division=1)
print("Classification Report (Testing Set):\n", classification_report_test)

# Function to recommend treatments for a specific grade
def recommend_treatments_for_grade(grade):
    # Extract treatments for the specific grade
    grade_treatments = df[df['Level'] == f'Grade {grade}']['Treatment Recommendation'].tolist()

    if grade_treatments:
        # Print treatment recommendations for the specific grade
        print(f"\nTreatment Recommendations for Grade {grade}:\n")
        for treatment in grade_treatments:
            print(treatment)
    else:
        print(f"\nNo treatment recommendations available for Grade {grade}.")

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

    if grade in grade_descriptions:
        print(grade_descriptions[grade])
    else:
        print("Invalid grade. Please enter a valid grade.")

# User input for grade
user_grade = input("Enter the grade for treatment recommendations (e.g., 0, 1, 2, ...): ")

# Print grade description
print_grade_description(user_grade)

# Recommend treatments for the user-specified grade
recommend_treatments_for_grade(user_grade)

# Calculate confusion matrix on the testing set
conf_matrix = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix (Testing Set):\n", conf_matrix)"""

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from imblearn.over_sampling import RandomOverSampler

# Load your dataset
df = pd.read_csv('Treatement dataset1.csv')

# Data preprocessing
# 1. Handle missing values (if any)
df_cleaned = df.dropna()

# 2. Convert categorical variables to one-hot encoding
df_encoded = pd.get_dummies(df_cleaned, columns=['Treatment Recommendation'], prefix='', prefix_sep='')

# Assuming 'Level' is the target variable and the rest are features
X = df_encoded.drop('Level', axis=1)
y = df_encoded['Level']

# 3. Apply RandomOverSampler to balance the data
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# 4. Split the resampled data into training, validation, and testing sets
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

# Save the trained model to a file
model_filename = 'random_forest_model_balanced.pkl'
joblib.dump(rf_model, model_filename)
print(f"Trained Random Forest model saved as {model_filename}")

# Classification Report for testing set
classification_report_test = classification_report(y_test, y_test_pred, zero_division=1)
print("Classification Report (Testing Set):\n", classification_report_test)