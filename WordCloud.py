import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re

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

# Ensure that the 'Review' and 'Emotion_Label' columns exist in the dataset
required_columns = ['Review', 'Emotion_Label']
missing_columns = [col for col in required_columns if col not in dataset.columns]

if missing_columns:
    raise ValueError(f"Dataset must contain the following columns: {', '.join(missing_columns)}")

# Function to clean text data
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Apply text cleaning
dataset['Cleaned_Review'] = dataset['Review'].apply(clean_text)

# Print sample data for verification
print("Sample cleaned reviews:")
print(dataset[['Review', 'Cleaned_Review']].head())

# Check the unique emotions in the dataset
emotions = dataset['Emotion_Label'].unique()
print("Unique emotions:", emotions)

# Generate and display word clouds for each emotion
for emotion in emotions:
    emotion_reviews = dataset[dataset['Emotion_Label'] == emotion]
    emotion_text = ' '.join(emotion_reviews['Cleaned_Review'])
    
    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=None).generate(emotion_text)
    
    # Display the word cloud
    plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Word Cloud of {emotion} Reviews')
    plt.axis('off')
    plt.show()
