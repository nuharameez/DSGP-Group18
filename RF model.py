import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix  # Corrected import
from imblearn.over_sampling import RandomOverSampler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the manually created dataset
df = pd.read_csv("Treatement dataset1.csv")

# Check for missing values
print(df.isnull().sum())

# Convert categorical columns to numerical using label encoding
label_encoder = LabelEncoder()
df['Treatment'] = label_encoder.fit_transform(df['Treatment'])

# Split data into features (X) and target (y)
X = df.drop(columns=['Severity'])
y = df['Severity']  # Set the target as 'Severity'

# Balance the dataset using oversampling
oversampler = RandomOverSampler(random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(X, y)

# Split resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train a Random Forest Classifier model
rf_model = RandomForestClassifier(n_estimators=15)
rf_model.fit(X_train, y_train)

# Evaluate the model
rf_predictions_train = rf_model.predict(X_train)
rf_accuracy_train = accuracy_score(y_train, rf_predictions_train)
print("Training Accuracy:", rf_accuracy_train)

# Evaluate the model
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print("Random Forest Classifier Accuracy:", rf_accuracy)
print("Classification Report:")
print(classification_report(y_test, rf_predictions))

# Compute confusion matrix
plot_confusion_matrix = confusion_matrix(y_test, rf_predictions)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(plot_confusion_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Save the trained model
joblib.dump(rf_model, 'random_forest_model.pkl')

# Load the saved model
loaded_model = joblib.load('random_forest_model.pkl')

# Define treatments for each severity level
treatments = df.groupby('Severity')['Treatment'].apply(lambda x: list(x.unique())).to_dict()

# Example: Use the trained model to predict treatments for a new severity level
new_data = np.array([[1]])  # Severity level
predicted_treatments = treatments[new_data[0, 0]]

# Print all treatments for the predicted severity level
print("Predicted Treatments for Severity Level", new_data[0, 0], ":")
for treatment in predicted_treatments:
    print("- ", label_encoder.inverse_transform([treatment])[0])