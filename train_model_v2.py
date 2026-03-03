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
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import StratifiedKFold
import unicodedata
import re

def detect_script(text):
    """Detect if text contains Hindi/Marathi characters"""
    return any('\u0900' <= char <= '\u097F' for char in text)

def clean_hindi_text(text):
    """Clean and normalize Hindi/Marathi text"""
    # Convert common variants
    variants = {
        'है?': 'है',
        'हैं?': 'हैं',
        'कौन-कौन': 'कौन',
        'क्या': 'किया'
    }
    for original, replacement in variants.items():
        text = text.replace(original, replacement)
    
    # Remove everything except Hindi characters and spaces
    text = re.sub(r'[^\u0900-\u097F\s]', '', text)
    return ' '.join(text.split())

def clean_english_text(text):
    """Clean and normalize English text"""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return ' '.join(text.split())
# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')

def normalize_text(text):
    """Wrapper to clean English text"""
    return clean_english_text(text)

def preprocess_hindi_text(text):
    """Wrapper to clean Hindi text"""
    return clean_hindi_text(text)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load and preprocess intents
with open('data/intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

# Print the number of intents to verify all are loaded
print(f"Loaded {len(intents['intents'])} intents")

# Initialize lists
words = []
classes = []
documents = []
ignore_chars = set('?!.,')

# Process each intent
for intent in intents['intents']:
    tag = intent['tag']
    classes.append(tag)
    for pattern in intent['patterns']:
        # Process based on script detection
        if detect_script(pattern):
            processed_pattern = clean_hindi_text(pattern)
        else:
            processed_pattern = clean_english_text(pattern)

        # Tokenize each word in the sentence
        word_list = nltk.word_tokenize(processed_pattern)
        words.extend(word_list)
        # Add to documents in our corpus
        documents.append((word_list, tag))

# Lemmatize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_chars]
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
def create_model(input_shape, output_shape):
    model = Sequential()
    model.add(Dense(256, input_shape=input_shape, activation='relu', kernel_regularizer=l2(0.0005)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.0005)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.0005)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(output_shape, activation='softmax'))

    adam = Adam(learning_rate=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model

# Get shapes for model
input_shape = (len(X_train[0]),)
output_shape = len(y_train[0])


# Train the model
print("Training the model...")

# Add early stopping to prevent overfitting
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=30,
    restore_best_weights=True,
    verbose=1
)

# Calculate class weights to handle imbalanced data
from sklearn.utils.class_weight import compute_class_weight
y_integers = np.argmax(y_train, axis=1)
class_weights = dict(enumerate(compute_class_weight('balanced', classes=np.unique(y_integers), y=y_integers)))

# Create k-fold cross-validation if classes have enough samples
class_counts = np.bincount(y_integers)
min_count = class_counts.min()
if min_count < 2:
    print(f"Not enough samples per class for cross-validation (min_count={min_count}). Skipping CV.")
    cv_scores = []
else:
    n_splits = min(3, int(min_count))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = []
    print("Starting cross-validation training...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_integers)):
        print(f"\nFold {fold + 1}/{n_splits}")
        # Split the data
        X_fold_train = np.array(X_train)[train_idx]
        y_fold_train = np.array(y_train)[train_idx]
        X_fold_val = np.array(X_train)[val_idx]
        y_fold_val = np.array(y_train)[val_idx]

        # Create a new model for each fold
        fold_model = create_model(input_shape, output_shape)

        # Train the model
        hist = fold_model.fit(X_fold_train, y_fold_train,
                        validation_data=(X_fold_val, y_fold_val),
                        epochs=100, batch_size=16, verbose=1,
                        callbacks=[early_stopping],
                        class_weight=class_weights)

        # Evaluate the model
        scores = fold_model.evaluate(X_fold_val, y_fold_val, verbose=0)
        cv_scores.append(scores[1])
        print(f"Fold {fold + 1} validation accuracy: {scores[1]*100:.2f}%")

    print("\nCross-validation complete!")
    print(f"Average validation accuracy: {np.mean(cv_scores)*100:.2f}%")

    print("\nTraining distribution:")
    unique, counts = np.unique(y_integers, return_counts=True)
    for i, (label, count) in enumerate(zip(unique, counts)):
        print(f"Class '{classes[label]}': {count} samples")

# Final training on all training data
print("\nTraining final model on all training data...")
# Create and train the final model
model = create_model(input_shape, output_shape)
hist = model.fit(np.array(X_train), np.array(y_train),
                validation_data=(np.array(X_test), np.array(y_test)),
                epochs=200, batch_size=32, verbose=1,
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
    
    # Preprocess the phrase
    if detect_script(phrase):
        processed_phrase = preprocess_hindi_text(phrase)
    else:
        processed_pattern = normalize_text(phrase)
        processed_phrase = processed_pattern
    
    # Tokenize and lemmatize
    words_in_phrase = nltk.word_tokenize(processed_phrase)
    words_in_phrase = [lemmatizer.lemmatize(word.lower()) for word in words_in_phrase]
    
    for se in words_in_phrase:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    
    # Get prediction
    results = model.predict(np.array([bag]))[0]
    
    # Get top predictions
    indices = np.argsort(results)[::-1]
    # Only show predictions with reasonable confidence
    indices = [i for i in indices if results[i] > 0.05][:3]
    
    print(f"Phrase: '{phrase}'")
    print("Top predictions:")
    if not indices:
        print("- No confident predictions (all probabilities low)")
    for i in indices:
        print(f"- {classes[i]} with probability {results[i]:.2f}")
    print()