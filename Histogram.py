import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
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

# Define a pipeline for text feature extraction and emotion classification
text_pipeline = Pipeline([
    ('vectorizer', CountVectorizer(ngram_range=(1, 2))),  # Include both unigrams and bigrams
    ('classifier', MultinomialNB())  # Using Naive Bayes for classification
])

# Train the model
text_pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = text_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy Score:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Plot histogram of Emotion Labels
sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))
sns.histplot(dataset['Emotion_Label'], bins=len(dataset['Emotion_Label'].unique()), kde=False, color='skyblue')
plt.title('Distribution of Emotion Labels', fontsize=16)
plt.xlabel('Emotion Label', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
