import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score

# Load the preprocessed dataset with emotion labels
file_path = r"E:\TYBSC CS\TCS INTERSHIP\preprocessed_emotions.csv"  # Updated to preprocessed file path
dataset = pd.read_csv(file_path)

# Strip leading and trailing whitespace from column names
dataset.columns = dataset.columns.str.strip()

# Ensure that the 'Review' and 'Emotion_Label' columns exist in the dataset
if 'Review' not in dataset.columns or 'Emotion_Label' not in dataset.columns:
    raise ValueError("Dataset must contain 'Review' and 'Emotion_Label' columns.")

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(dataset['Review'], dataset['Emotion_Label'], test_size=0.2, random_state=42)

# Define a list of hyperparameters to explore
hyperparameters = [
    {'alpha': 1.0},  # Default value for MultinomialNB
    {'alpha': 0.5},
    {'alpha': 2.0}
]

# Train models with different hyperparameters
models = []
for params in hyperparameters:
    model = make_pipeline(
        CountVectorizer(ngram_range=(1, 2)),  # Include both unigrams and bigrams
        MultinomialNB(**params)
    )
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
print(classification_report(y_test, y_pred_best))
