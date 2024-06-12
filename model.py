import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Load dataset
data = pd.read_csv('winequality-red.csv')

# Clipping outliers using IQR method
def iqr_clipping(data, cols):
    for col in cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        low_lim = Q1 - 1.5 * IQR
        upp_lim = Q3 + 1.5 * IQR
        data[col] = data[col].clip(lower=low_lim, upper=upp_lim)
    return data

cols = ['density', 'fixed acidity', 'volatile acidity', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'citric acid']
data = iqr_clipping(data, cols)

# Define features and target
X = data.drop(columns=['quality'], axis=1)
y = data['quality']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handling class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Binarize the target: good (quality >= 7) and bad (quality < 7)
y_resampled = y_resampled.apply(lambda y_value: 1 if y_value >= 7 else 0)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, random_state=42, test_size=0.2)

# Train Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, y_test)
print('The test data accuracy is', test_data_accuracy)

# Save the model and scaler to files
pickle.dump(model, open('redwine.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

