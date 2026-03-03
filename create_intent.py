# Create create_intents.py
import json

# Define intents data
intents_data = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["Hi", "Hello", "Hey", "Good morning"],
            "responses": ["Hello!", "Hi there, how can I help?"]
        },
        {
            "tag": "admission",
            "patterns": ["How to apply for admission?", "What are admission requirements?"],
            "responses": ["You can apply online through our portal. Requirements include 12th grade with 60% marks."]
        },
        {
            "tag": "fees",
            "patterns": ["What is the fee structure?", "How much is the tuition fee?"],
            "responses": ["The annual fee for B.Tech is ₹1,20,000. Hostel fee is ₹45,000 per year."]
        },
        {
            "tag": "courses",
            "patterns": ["What courses do you offer?", "Tell me about available courses"],
            "responses": ["We offer B.Tech in Computer Science, Mechanical, Civil, and Electrical Engineering."]
        },
        {
            "tag": "goodbye",
            "patterns": ["Bye", "See you later", "Goodbye"],
            "responses": ["Goodbye!", "Have a nice day!"]
        }
    ]
}

# Save with UTF-8 encoding
with open('data/intents.json', 'w', encoding='utf-8') as file:
    json.dump(intents_data, file, ensure_ascii=False, indent=4)

print("Intents file created successfully!")