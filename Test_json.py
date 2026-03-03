# Create test_json.py
import json

# Try different encodings to read the file
encodings = ['utf-8', 'utf-16', 'iso-8859-1', 'cp1252']

for encoding in encodings:
    try:
        with open('data/intents.json', 'r', encoding=encoding) as file:
            data = json.load(file)
        print(f"Successfully read with {encoding} encoding")
        print(f"Number of intents: {len(data['intents'])}")
        break
    except Exception as e:
        print(f"Failed with {encoding} encoding: {e}")