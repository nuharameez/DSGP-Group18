import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load dataset from file
def load_dataset(filename):
    df = pd.read_csv(filename)
    return df

# Function to encode categorical features
def encode_categorical_features(df):
    global label_encoder  # Declare label_encoder as global
    label_encoder = LabelEncoder()  # Define LabelEncoder
    df['Treatment'] = label_encoder.fit_transform(df['Treatment'])
    return df

# Function to split data into features and target
def split_data(df):
    X = df.drop(columns=['Treatment'])
    y = df['Severity']
    return X, y

# Function to balance dataset
def balance_dataset(X, y):
    oversampler = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(X, y)
    return X_resampled, y_resampled

# Function to split data into training and testing sets
def split_train_test_data(X_resampled, y_resampled):
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Function to train model
def train_model(X_train, y_train):
    rf_model = RandomForestClassifier(n_estimators=15)
    rf_model.fit(X_train, y_train)
    return rf_model

# Function to evaluate model
def evaluate_model(model, X_test, y_test):
    rf_predictions = model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_predictions)
    print("Random Forest Classifier Accuracy:", rf_accuracy)
    print("Classification Report:")
    print(classification_report(y_test, rf_predictions))
    plot_confusion_matrix = confusion_matrix(y_test, rf_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(plot_confusion_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# Function to save model
def save_model(model, filename):
    joblib.dump(model, filename)

# Function to load model
def load_model(filename):
    loaded_model = joblib.load(filename)
    return loaded_model

# Function to predict treatments for new data
def predict_treatments(loaded_model, new_data, severity_treatments, label_encoder):
    if new_data['Severity'][0] not in severity_treatments:
        print("Error: Predicted severity level is not present in the dataset.")
    else:
        predicted_severity = loaded_model.predict(new_data)
        treatments_encoded = severity_treatments[predicted_severity[0]]
        treatments_inverse = label_encoder.inverse_transform(treatments_encoded)
        treatments_inverse = [str(treatment) for treatment in treatments_inverse]  # Convert treatments to string
        print("Predicted Treatments for Severity Level", predicted_severity[0], ":")
        for treatment in treatments_inverse:
            print("- ", treatment)

# Load dataset
df = load_dataset("Treatment dataset1.csv")

# Encode categorical features
df = encode_categorical_features(df)  # Call the function without passing label_encoder

# Split data into features and target
X, y = split_data(df)

# Balance dataset
X_resampled, y_resampled = balance_dataset(X, y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = split_train_test_data(X_resampled, y_resampled)

# Train model
rf_model = train_model(X_train, y_train)

# Evaluate model
evaluate_model(rf_model, X_test, y_test)

# Save model
save_model(rf_model, 'random_forest_model.pkl')

# Load model
loaded_model = load_model('random_forest_model.pkl')

# Define treatments for each severity level
severity_treatments = df.groupby('Severity')['Treatment'].apply(lambda x: list(x.unique())).to_dict()

# Example: Use the trained model to predict treatments for a new severity level
new_data = pd.DataFrame([[0]], columns=['Severity'])  # Severity level

# Predict treatments
predict_treatments(loaded_model, new_data, severity_treatments, label_encoder)

