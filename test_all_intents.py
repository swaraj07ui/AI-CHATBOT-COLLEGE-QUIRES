# Create test_all_intents.py
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# Load model and data
model = load_model('models/chatbot_model.h5')
with open('data/intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)
words = pickle.load(open('data/words.pkl', 'rb'))
classes = pickle.load(open('data/classes.pkl', 'rb'))

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to clean up sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Function to create bag of words
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Test each intent
print("Testing all intents...\n")

for intent in intents['intents']:
    tag = intent['tag']
    patterns = intent['patterns']
    
    print(f"Intent: {tag}")
    print("Patterns:")
    
    for pattern in patterns:
        # Create bag of words
        bag = bag_of_words(pattern)
        
        # Get prediction
        results = model.predict(np.array([bag]))[0]
        result_index = np.argmax(results)
        result_tag = classes[result_index]
        result_prob = results[result_index]
        
        print(f"  Pattern: '{pattern}'")
        print(f"  Predicted: {result_tag} (prob: {result_prob:.2f})")
        
        if result_tag != tag:
            print(f"  ❌ MISMATCH! Expected: {tag}")
        else:
            print(f"  ✅ Correct")
        print()