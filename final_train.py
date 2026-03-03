import numpy as np
import json
import pickle
import random
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split

# Download necessary NLTK data
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

# Process each intent
for intent in data['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Save the words and classes
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

# Shuffle the features and turn into np.array
random.shuffle(training)
training = np.array(training, dtype=object)

# Create train and test lists
X = list(training[:, 0])
y = list(training[:, 1])

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = Sequential()
model.add(Dense(128, input_shape=(len(X_train[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]), activation='softmax'))

# Compile the model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model with validation
hist = model.fit(
    np.array(X_train), 
    np.array(y_train), 
    validation_data=(np.array(X_test), np.array(y_test)),
    epochs=300, 
    batch_size=5, 
    verbose=1
)

# Save the model
model.save('models/chatbot_model.h5', hist)

# Print final metrics
print(f"Final training accuracy: {hist.history['accuracy'][-1]}")
print(f"Final validation accuracy: {hist.history['val_accuracy'][-1]}")

# Test with specific phrases
test_phrases = [
    "How to apply for admission?",
    "What is the fee structure?",
    "What courses do you offer?"
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