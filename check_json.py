import json

try:
    with open('data/intents.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    print("JSON file loaded successfully!")
    print(f"Number of intents: {len(data['intents'])}")
    
    # Print the first intent
    if len(data['intents']) > 0:
        first_intent = data['intents'][0]
        print(f"\nFirst intent tag: {first_intent['tag']}")
        print(f"Patterns: {first_intent['patterns'][:2]}...")  # Show first 2 patterns
        print(f"Responses: {first_intent['responses'][:2]}...")  # Show first 2 responses
    
except Exception as e:
    print(f"Error loading JSON file: {e}")