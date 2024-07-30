import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset from a CSV file
file_path = r"E:\TYBSC CS\TCS INTERSHIP\preprocessed_emotions.csv"  # Replace with your actual CSV file path
try:
    dataset = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Strip leading and trailing whitespace from column names
dataset.columns = dataset.columns.str.strip()

# Assume 'Features' are all columns except 'Emotion_Label' which is the column to predict
target_column = 'Emotion_Label'  # Replace with the actual target column name

# Check if the target column exists in the dataset
if target_column not in dataset.columns:
    raise ValueError(f"Dataset must contain the '{target_column}' column.")

# Separate features and target variable
X = dataset.drop(columns=[target_column])
y = dataset[target_column]

# Handle missing values if any
X = X.fillna(X.mean())
y = y.fillna(y.mode()[0])  # Use mode for categorical target variable

# Encode categorical target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a list of hyperparameters to explore
hyperparameters = [
    {'C': 1.0},  # Default value for LogisticRegression
    {'C': 0.5},
    {'C': 2.0}
]

# Train models with different hyperparameters
models = []
for params in hyperparameters:
    model = make_pipeline(StandardScaler(), LogisticRegression(**params, max_iter=1000))
    model.fit(X_train, y_train)
    models.append(model)

# Evaluate model performance
accuracy_scores = []
for model in models:
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

# Compare performance
best_model_index = accuracy_scores.index(max(accuracy_scores))
best_model = models[best_model_index]
best_params = hyperparameters[best_model_index]
best_accuracy = accuracy_scores[best_model_index]

# Print results
print("Model Performances:")
for i, params in enumerate(hyperparameters):
    print(f"Model {i+1} Hyperparameters:", params)
    print(f"Model {i+1} Accuracy:", accuracy_scores[i])
    print()

print("Best Model Hyperparameters:", best_params)
print("Best Model Accuracy:", best_accuracy)

# Print classification report for the best model
y_pred_best = best_model.predict(X_test)
print("\nClassification Report for Best Model:")
print(classification_report(y_test, y_pred_best, target_names=label_encoder.classes_))
