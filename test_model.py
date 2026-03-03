import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load the model and data
model = load_model('models/chatbot_model.h5')
words = pickle.load(open('data/words.pkl', 'rb'))
classes = pickle.load(open('data/classes.pkl', 'rb'))

# Test with a specific phrase
test_phrase = "How to apply for admission?"

# Create bag of words
def bag_of_words(s):
    bag = [0] * len(words)
    s_words = s.lower().split()
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag)

# Test the model
test_bag = bag_of_words(test_phrase)
print(f"Test phrase: '{test_phrase}'")
print(f"Bag of words (non-zero indices): {[i for i, val in enumerate(test_bag) if val == 1]}")

# Get prediction
results = model.predict(np.array([test_bag]))[0]
print(f"Model prediction: {results}")

# Get the highest probability
result_index = np.argmax(results)
result_tag = classes[result_index]
result_prob = results[result_index]

print(f"Predicted tag: {result_tag} with probability {result_prob}")

if result_prob > 0.25:
    print("Intent recognized successfully!")
else:
    print("Intent not recognized with high confidence.")