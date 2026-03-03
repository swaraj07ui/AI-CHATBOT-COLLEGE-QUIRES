import json
import numpy as np
import nltk
import pickle
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Read the intents file
with open('data/intents.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Initialize lists
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

# Process each intent
for intent in data['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word in the sentence
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        # Add the documents in the corpus
        documents.append((word_list, intent['tag']))
        # Add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(list(set(words)))

# Sort classes
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
import random
random.shuffle(training)
training = np.array(training, dtype=object)

# Create train and test lists
X = list(training[:, 0])
y = list(training[:, 1])

# Save the processed data
np.save('data/X.npy', X)
np.save('data/y.npy', y)

print("Data preprocessed and saved.")
print(f"Vocabulary size: {len(words)}")
print(f"Number of classes: {len(classes)}")