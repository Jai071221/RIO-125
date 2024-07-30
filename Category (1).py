import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nrclex import NRCLex

# Load the dataset from a CSV file
file_path = r"E:\TYBSC CS\TCS INTERSHIP\reviews.csv"  # Replace with your actual CSV file path
try:
    dataset = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Strip leading and trailing whitespace from column names
dataset.columns = dataset.columns.str.strip()

# Ensure that the 'Review' and 'Rating' columns exist in the dataset
if 'Review' not in dataset.columns or 'Rating' not in dataset.columns:
    raise ValueError("Dataset must contain 'Review' and 'Rating' columns.")

# Emotion analysis implementation
def perform_emotion_analysis(text):
    emotions = NRCLex(str(text)).top_emotions
    if emotions:
        # Get the emotion with the highest score
        dominant_emotion = max(emotions, key=lambda x: x[1])[0]
    else:
        dominant_emotion = "Neutral"
    return dominant_emotion

# Apply emotion analysis
dataset['Dominant_Emotion'] = dataset['Review'].apply(perform_emotion_analysis)

# Ensure the 'Rating' column is numeric
dataset['Rating'] = pd.to_numeric(dataset['Rating'], errors='coerce')

# Drop rows with missing values in 'Rating' or 'Dominant_Emotion'
dataset.dropna(subset=['Rating', 'Dominant_Emotion'], inplace=True)

# Box Plot of Ratings by Dominant Emotion Category
sns.set_style("whitegrid")
plt.figure(figsize=(12, 8))
sns.boxplot(x='Dominant_Emotion', y='Rating', data=dataset, palette='Set2')
plt.title('Ratings by Dominant Emotion Category', fontsize=16)
plt.xlabel('Dominant Emotion', fontsize=14)
plt.ylabel('Rating', fontsize=14)
plt.xticks(fontsize=12, rotation=45)
plt.yticks(fontsize=12)
plt.show()
