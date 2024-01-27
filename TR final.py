# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib  # Import joblib for saving the model

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

# Calculate accuracy on the training set
y_train_pred = rf_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy:.2f}")

# Classification Report for training set
classification_report_train = classification_report(y_train, y_train_pred, zero_division=1)
print("Classification Report (Training Set):\n", classification_report_train)

# Save the trained model to a file
model_filename = 'random_forest_model.joblib'
joblib.dump(rf_model, model_filename)
print(f"Trained Random Forest model saved as {model_filename}")

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Calculate accuracy on the testing set
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Testing Accuracy: {test_accuracy:.2f}")

# Evaluate the model
classification_report_output = classification_report(y_test, y_pred, zero_division=1)

# Print the results
print("Classification Report (Testing Set):\n", classification_report_output)


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

# User input for grade
user_grade = input("Enter the grade for treatment recommendations (e.g., 0, 1, 2, ...): ")

# Recommend treatments for the user-specified grade
recommend_treatments_for_grade(user_grade)

