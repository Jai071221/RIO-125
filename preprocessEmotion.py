import pandas as pd
from textblob import TextBlob

# Load the dataset
file_path = r"E:\TYBSC CS\TCS INTERSHIP\reviews.csv"  # Replace with your actual CSV file path
dataset = pd.read_csv(file_path)

# Strip leading and trailing whitespace from column names
dataset.columns = dataset.columns.str.strip()

# Check if the 'Review' column exists
if 'Review' not in dataset.columns:
    raise ValueError("Dataset must contain 'Review' column.")

# Define a function to assign emotion labels based on some criteria or emotion detection model
def assign_emotion(text):
    # For demonstration, we'll use a simple approach based on sentiment analysis
    # In a real scenario, use an emotion detection library or model
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0.5:
        return 'Joy'
    elif polarity > 0:
        return 'Happiness'
    elif polarity == 0:
        return 'Neutral'
    elif polarity > -0.5:
        return 'Sadness'
    else:
        return 'Anger'

# Apply the emotion assignment function
dataset['Emotion_Label'] = dataset['Review'].apply(assign_emotion)

# Save the updated dataset with emotion labels
preprocessed_file_path = r"E:\TYBSC CS\TCS INTERSHIP\preprocessed_emotions.csv"
dataset.to_csv(preprocessed_file_path, index=False)

print("Dataset with emotion labels saved successfully.")
