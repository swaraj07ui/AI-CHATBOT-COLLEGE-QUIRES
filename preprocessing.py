import json
import nltk
import numpy as np
import pickle
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder

nltk.download('punkt_tab')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

def preprocess_data(intents_file):
    with open('data/intents.json', 'r', encoding='utf-8') as file:
     data = json.load(file)
    
    words = []
    classes = []
    documents = []
    
    for intent in data['intents']:
        for pattern in intent['patterns']:
            tokens = nltk.word_tokenize(pattern)
            words.extend(tokens)
            documents.append((tokens, intent['tag']))
            
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
    
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word.isalnum()]
    words = sorted(set(words))
    classes = sorted(set(classes))
    
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
    
    training = np.array(training, dtype=object)
    X = list(training[:, 0])
    y = list(training[:, 1])
    
    return X, y, words, classes

if __name__ == "__main__":
    X, y, words, classes = preprocess_data('data/intents.json')
    np.save('data/X.npy', X)
    np.save('data/y.npy', y)
    np.save('data/words.npy', words)
    np.save('data/classes.npy', classes)
    with open('data/words.pkl', 'wb') as f:
        pickle.dump(words, f)
    with open('data/classes.pkl', 'wb') as f:
        pickle.dump(classes, f)
        
  