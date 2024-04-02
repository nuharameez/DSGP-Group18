from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import pandas as pd

# Load the combined dataset
df = pd.read_csv('Treatement dataset2.csv')

# Split the data into features (X) and target variable (y)
X = df.drop(['Severity'], axis=1)  # Features
y = df['Severity']  # Target variable

# Apply one-hot encoding to categorical variables
# Here, we're assuming Treatment is the only categorical variable
ct = ColumnTransformer([('encoder', OneHotEncoder(), ['Treatment', 'Age_Category', 'Gender'])], remainder='passthrough')
X_encoded = ct.fit_transform(X)

# Split the encoded data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Initialize the random forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Save the trained model
joblib.dump(clf, 'rf_model.pkl')

import joblib

# Function to get treatments for a given severity level
def get_treatments(severity_level, age_category, gender, model):
    treatments = df[(df['Severity'] == severity_level) & (df['Age_Category'] == age_category) & (df['Gender'] == gender)]['Treatment']
    return treatments.tolist()

# Load the trained model
model = joblib.load('rf_model.pkl')

# Function to recommend treatments based on user inputs
def recommend_treatments():
    severity_input = input("Enter severity level (0 to 4) or 'exit' to quit: ")
    if severity_input.lower() == 'exit':
        print("Exiting...")
        return
    try:
        severity_level = int(severity_input)
        if severity_level < 0 or severity_level > 4:
            print("Invalid severity level. Please enter a number between 0 and 4.")
            return
        age_category = input("Enter age category (<18, 18-35, 35-60, >60): ")
        gender = input("Enter gender (Male/Female): ")
        severity_treatments = get_treatments(severity_level, age_category, gender, model)
        if severity_treatments:
            print(f"Treatments for severity level {severity_level}, age category {age_category}, and gender {gender}:")
            for treatment in severity_treatments:
                print("-", treatment)
        else:
            print(f"No treatments found for severity level {severity_level}, age category {age_category}, and gender {gender}.")
    except ValueError:
        print("Invalid input. Please enter a valid severity level (0 to 4) or 'exit' to quit.")

# Test the function repeatedly until 'exit' is entered
while True:
    recommend_treatments()
    decision = input("Do you want to recommend treatments again? (yes/no): ")
    if decision.lower() == 'no':
        print("Exiting the loop...")
        break

# Predict on the test set
y_pred = clf.predict(X_test)

# Print Training Accuracy
train_predictions = clf.predict(X_train)
train_accuracy = accuracy_score(y_train, train_predictions)
print("Training Accuracy:", train_accuracy)

# Print the classification report
print(classification_report(y_test, y_pred))