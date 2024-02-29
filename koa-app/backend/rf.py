import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
df = pd.read_csv(r"C:\Users\multi\Desktop\All Folders\Treatement dataset1.csv")

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
