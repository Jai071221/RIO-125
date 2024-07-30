import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import string
from textblob import TextBlob

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Dataset loading
file_path = r"E:\TYBSC CS\TCS INTERSHIP\reviews.csv"
output_file_path = r"E:\TYBSC CS\TCS INTERSHIP\preprocessed_reviews.csv"  # Path for the output CSV

try:
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Strip leading and trailing whitespace from column names
df.columns = df.columns.str.strip()

# Print the cleaned column names to verify
print("Cleaned column names:")
print(df.columns)

# Ensure the 'Review' and 'Rating' columns exist in the dataset
if 'Review' not in df.columns or 'Rating' not in df.columns:
    raise ValueError("Dataset must contain 'Review' and 'Rating' columns.")

print("Columns check passed.")

# Text preprocessing
def preprocess_text(text):
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = []
    for word, tag in pos_tags:
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_tokens.append(lemmatizer.lemmatize(word, pos))
    table = str.maketrans('', '', string.punctuation)
    no_punctuation_tokens = [token.translate(table) for token in lemmatized_tokens]
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in no_punctuation_tokens if word.lower() not in stop_words]
    preprocessed_text = ' '.join(filtered_tokens)
    return preprocessed_text

# Sentiment analysis implementation
def perform_sentiment_analysis(text):
    analysis = TextBlob(str(text))
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        sentiment_label = "Positive"
    elif polarity == 0:
        sentiment_label = "Neutral"
    else:
        sentiment_label = "Negative"
    return sentiment_label, polarity

# Integration of text preprocessing and sentiment analysis results into the dataset
try:
    print("Starting text preprocessing...")
    df['Preprocessed_Review'] = df['Review'].apply(preprocess_text)
    print("Text preprocessing completed.")
    
    print("Starting sentiment analysis...")
    df['Sentiment_Label'], df['Sentiment_Polarity'] = zip(*df['Preprocessed_Review'].apply(perform_sentiment_analysis))
    print("Sentiment analysis completed.")
except Exception as e:
    print(f"Error during text processing: {e}")
    exit()

# Ensure the 'Rating' column is numeric
df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')

# Drop rows with missing values in 'Sentiment_Polarity' or 'Rating'
df.dropna(subset=['Sentiment_Polarity', 'Rating'], inplace=True)
print("Data cleaning completed.")

# Save the preprocessed data to a new CSV file
try:
    df.to_csv(output_file_path, index=False)
    print(f"Preprocessed data saved to {output_file_path}.")
except Exception as e:
    print(f"Error saving preprocessed data: {e}")

# Print the first few rows of the processed dataset to verify
print(df.head())
