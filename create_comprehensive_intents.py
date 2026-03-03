# Create create_comprehensive_intents.py
import json

# Define comprehensive intents data
intents_data = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": [
                "Hi", "Hello", "Hey", "Good morning", "Good afternoon", "Good evening",
                "Greetings", "How are you", "What's up", "How do you do"
            ],
            "responses": [
                "Hello!", "Hi there!", "Hey!", "Good to see you!",
                "Greetings!", "How can I help you today?"
            ]
        },
        {
            "tag": "admission",
            "patterns": [
                "How to apply for admission?", "What are admission requirements?",
                "Tell me about admission process", "How can I get admission?",
                "What is the admission procedure?", "Admission process",
                "How do I apply?", "Application process", "How to enroll?"
            ],
            "responses": [
                "You can apply online through our portal. Requirements include 12th grade with 60% marks.",
                "To apply for admission, visit our website and fill the online form. You need 60% in 12th grade.",
                "The admission process is online. You need 60% marks in your 12th grade to be eligible."
            ]
        },
        {
            "tag": "fees",
            "patterns": [
                "What is the fee structure?", "How much is the tuition fee?",
                "Tell me about fees", "Fee details", "What are the charges?",
                "How much do I need to pay?", "Tuition fees", "Cost of education"
            ],
            "responses": [
                "The annual fee for B.Tech is ₹1,20,000. Hostel fee is ₹45,000 per year.",
                "Tuition fee for B.Tech is ₹1,20,000 per year. Hostel accommodation costs ₹45,000 annually.",
                "The total fee structure is ₹1,20,000 for tuition and ₹45,000 for hostel per year."
            ]
        },
        {
            "tag": "courses",
            "patterns": [
                "What courses do you offer?", "Tell me about available courses",
                "What programs are available?", "List of courses", "What can I study?",
                "What are the academic programs?", "What subjects do you teach?"
            ],
            "responses": [
                "We offer B.Tech in Computer Science, Mechanical, Civil, and Electrical Engineering.",
                "Our college offers B.Tech programs in Computer Science, Mechanical, Civil, and Electrical Engineering.",
                "Available courses include B.Tech in Computer Science, Mechanical Engineering, Civil Engineering, and Electrical Engineering."
            ]
        },
        {
            "tag": "goodbye",
            "patterns": [
                "Bye", "See you later", "Goodbye", "Take care", "Have a nice day",
                "See you", "Farewell", "I'm leaving"
            ],
            "responses": [
                "Goodbye!", "Have a nice day!", "See you later!",
                "Take care!", "Farewell!"
            ]
        }
    ]
}

# Save with UTF-8 encoding
with open('data/intents.json', 'w', encoding='utf-8') as file:
    json.dump(intents_data, file, ensure_ascii=False, indent=4)

print("Comprehensive intents file created successfully!")