# Create retrain_with_more_epochs.py
import numpy as np
import json
import pickle
import random
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Load intents
with open('data/intents.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Initialize lists
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

# Process intents
for intent in data['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize words
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

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

# Shuffle and convert to numpy array
random.shuffle(training)
training = np.array(training, dtype=object)

# Split into features and labels
X = list(training[:, 0])
y = list(training[:, 1])

# Build model with more capacity
model = Sequential()
model.add(Dense(256, input_shape=(len(X[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y[0]), activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model with more epochs
print("Training model...")
model.fit(np.array(X), np.array(y), epochs=500, batch_size=5, verbose=1)

# Save model
model.save('models/chatbot_model.h5')
print("Model saved!")

# Test with sample phrases
test_phrases = [
    "How to apply for admission?",
    "What is the fee structure?",
    "What courses do you offer?",
    "Hi there",
    "Goodbye"
]

print("\nTesting with sample phrases:")
for phrase in test_phrases:
    # Create bag of words
    bag = [0] * len(words)
    s_words = phrase.lower().split()
    s_words = [lemmatizer.lemmatize(word.lower()) for word in s_words]
    
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    
    # Get prediction
    results = model.predict(np.array([bag]))[0]
    result_index = np.argmax(results)
    result_tag = classes[result_index]
    result_prob = results[result_index]
    
    print(f"Phrase: '{phrase}'")
    print(f"Predicted tag: {result_tag} with probability {result_prob}")
    print()