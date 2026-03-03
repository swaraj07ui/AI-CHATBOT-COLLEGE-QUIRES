import json
import numpy as np
import pickle
import random
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
import re
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import StratifiedKFold
import unicodedata

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer
def normalize_text(text):
    """Normalize text by removing diacritics, lowercasing, and basic cleaning"""
    # Remove diacritics
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    # Convert to lowercase
    text = text.lower()
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^a-z0-9\s\?\!\.\,]', '', text)
    return text

def preprocess_hindi_text(text):
    """Special preprocessing for Hindi/Marathi text"""
    # Split on spaces and remove punctuation
    words = re.findall(r'\w+', text)
    # Join back with spaces
    return ' '.join(words)
lemmatizer = WordNetLemmatizer()

# Load intents file from data directory
with open('data/intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

# Print the number of intents to verify all are loaded
print(f"Loaded {len(intents['intents'])} intents")

# Initialize lists
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

# Process each intent
for intent in intents['intents']:
    tag = intent['tag']
    classes.append(tag)
    
    for pattern in intent['patterns']:
        # Tokenize each word in the sentence
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        # Add to documents in our corpus
        documents.append((word_list, tag))

# Lemmatize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(set(words))

# Sort classes
classes = sorted(set(classes))

# Save words and classes
with open('data/words.pkl', 'wb') as f:
    pickle.dump(words, f)

with open('data/classes.pkl', 'wb') as f:
    pickle.dump(classes, f)

# Create training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    
    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)
    
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])

# Shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training, dtype=object)

# Create train and test lists
X = list(training[:, 0])
y = list(training[:, 1])

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = Sequential()
model.add(Dense(512, input_shape=(len(X_train[0]),), activation='relu', kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(len(y_train[0]), activation='softmax'))

# Compile model
adam = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# Train the model
print("Training the model...")
# Add early stopping to prevent overfitting
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

# Calculate class weights to handle imbalanced data
from sklearn.utils.class_weight import compute_class_weight
y_integers = np.argmax(y_train, axis=1)
class_weights = dict(enumerate(compute_class_weight('balanced', classes=np.unique(y_integers), y=y_integers)))

hist = model.fit(np.array(X_train), np.array(y_train), 
                 validation_data=(np.array(X_test), np.array(y_test)),
                epochs=500, batch_size=32, verbose=1,
                callbacks=[early_stopping],
                class_weight=class_weights)

# Save the model
model.save('models/chatbot_model.h5', hist)
print("Model created and saved.")

# Evaluate the model
print("Evaluating the model...")
loss, accuracy = model.evaluate(np.array(X_test), np.array(y_test), verbose=0)
print(f"Model Accuracy: {accuracy*100:.2f}%")

# Test with sample phrases
test_phrases = [
    "What courses are available?",
    "कौन-कौन से कोर्स उपलब्ध हैं?",
    "आपके पास कौन से प्रोग्राम हैं?",
    "Is there any scholarship?",
    "क्या कोई छात्रवृत्ति है?",
    "What is the placement record?",
    "प्लेसमेंट कैसा है?",
    "Is hostel facility available?",
    "क्या हॉस्टल की सुविधा है?"
]

print("\nTesting with sample phrases:")
for phrase in test_phrases:
    # Create bag of words
    bag = [0] * len(words)
    
    # Handle Hindi/Marathi text properly
    # Split the phrase into words
    if any(char in phrase for char in ['\u0900', '\u097F']):  # Check for Hindi/Marathi characters
        # For Hindi/Marathi, we'll split by spaces and remove punctuation
        words_in_phrase = re.findall(r'\w+', phrase)
    else:
        words_in_phrase = nltk.word_tokenize(phrase)
    
    words_in_phrase = [lemmatizer.lemmatize(word.lower()) for word in words_in_phrase]
    
    for se in words_in_phrase:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    
    # Get prediction
    results = model.predict(np.array([bag]))[0]
    result_index = np.argmax(results)
    result_tag = classes[result_index]
    result_prob = results[result_index]
    
    print(f"Phrase: '{phrase}'")
    print(f"Predicted tag: {result_tag} with probability {result_prob:.2f}")
    print()