import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import string
from nrclex import NRCLex
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Dataset loading
file_path = "e:/TYBSC CS/TCS INTERSHIP/reviews.csv"  # Updated to actual file path
try:
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Strip leading and trailing whitespace from column names
df.columns = df.columns.str.strip()

# Ensure the 'Review' column exists in the dataset
if 'Review' not in df.columns:
    raise ValueError("Dataset must contain 'Review' column.")

print("Columns check passed.")

# Text preprocessing function
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

# Emotion analysis function using NRCLex
def perform_emotion_analysis(text):
    emotion_analyzer = NRCLex(text)
    emotions = emotion_analyzer.affect_frequencies
    # Choose the emotion with the highest frequency
    if emotions:
        emotion_label = max(emotions, key=emotions.get)
    else:
        emotion_label = "Neutral"
    return emotion_label

# Apply text preprocessing and emotion analysis
try:
    print("Starting text preprocessing...")
    df['Preprocessed_Review'] = df['Review'].apply(preprocess_text)
    print("Text preprocessing completed.")
    
    print("Starting emotion analysis...")
    df['Emotion_Label'] = df['Preprocessed_Review'].apply(perform_emotion_analysis)
    print("Emotion analysis completed.")
except Exception as e:
    print(f"Error during processing: {e}")
    exit()

# Convert emotion labels to numerical values
label_encoder = LabelEncoder()
df['Emotion_Label'] = label_encoder.fit_transform(df['Emotion_Label'])

# Prepare data for deep learning model
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['Preprocessed_Review'])
X = tokenizer.texts_to_sequences(df['Preprocessed_Review'])
X = pad_sequences(X)
y = df['Emotion_Label']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the deep learning model architecture
embedding_dim = 100
max_sequence_length = X.shape[1]
num_classes = len(label_encoder.classes_)

model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=embedding_dim, input_length=max_sequence_length))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(units=128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(units=num_classes, activation='softmax'))  # Number of classes for emotions

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.1, callbacks=[early_stopping])

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

# Save the preprocessed dataset with emotion labels
output_file_path = "e:/TYBSC CS/TCS INTERSHIP/preprocessed_reviews_with_emotions.csv"
df.to_csv(output_file_path, index=False)
print(f"Preprocessed dataset saved to {output_file_path}")
