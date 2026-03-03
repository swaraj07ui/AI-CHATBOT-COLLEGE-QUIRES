import random
import json
import pickle
import numpy as np
import nltk
import re
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from voice_assistant import VoiceAssistant
from ai_integration import AIIntegration
from erp_integration import ERPIntegration

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load the trained model
model = load_model('models/chatbot_model.h5')

# Load words and classes
with open('data/words.pkl', 'rb') as f:
    words = pickle.load(f)

with open('data/classes.pkl', 'rb') as f:
    classes = pickle.load(f)

# Load intents file from data directory
with open('data/intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

# Initialize components
voice_assistant = VoiceAssistant()
ai_integration = AIIntegration()
erp_integration = ERPIntegration()

# Context memory
context = {}

def clean_up_sentence(sentence):
    """Tokenize and lemmatize the input sentence with improved Hindi/Marathi support"""
    # Check if the sentence contains Hindi or Marathi characters
    if any('\u0900' <= char <= '\u097F' for char in sentence):  # Hindi/Marathi Unicode range
        # For Hindi/Marathi, split by spaces and preserve full words
        words = sentence.split()
        # Clean each word but preserve the full word form
        sentence_words = [re.sub(r'[?!.,।]', '', word) for word in words]
    else:
        # For English, use NLTK tokenizer and lemmatize
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    
    return sentence_words

def bag_of_words(sentence):
    """Create a bag of words from the input sentence"""
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    """Predict the intent class of the input sentence"""
    p = bag_of_words(sentence)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    # Sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(ints, intents_json, user_id):
    """Get a response based on the predicted intent"""
    if not ints:
        return "I'm sorry, I don't understand that."
    
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    
    # Check if there's context for this user
    if user_id in context and context[user_id]:
        context_tag = context[user_id]
        for intent in list_of_intents:
            if intent['tag'] == context_tag:
                result = random.choice(intent['responses'])
                # Clear context after using it
                del context[user_id]
                return result
    
    for i in list_of_intents:
        if i['tag'] == tag:
            # Filter responses based on language
            responses = [r for r in i['responses'] if not any(char in '\u0900-\u097F' for char in r)]  # Get only English responses
            if responses:  # If there are English responses available
                result = random.choice(responses)
                # Set context if specified
                if 'context_set' in i:
                    context[user_id] = i['context_set']
                return result
            # Fallback to any response if no English responses found
            result = random.choice(i['responses'])
            if 'context_set' in i:
                context[user_id] = i['context_set']
            return result
    
    # If no matching intent is found, use AI integration
    try:
        # Get context for AI
        user_context = ""
        if user_id in context:
            user_context = "\n".join([f"{c['query']}: {c['response']}" for c in context[user_id][-3:]])
        
        # Get AI response
        ai_response = ai_integration.get_openai_response(
            f"{user_context}\nUser: {tag}", 
            context=user_context
        )
        return ai_response
    except:
        return "I'm still learning about that topic. Could you ask something else?"

def check_erp_query(query, user_id):
    """Check if the query is related to ERP data"""
    query_lower = query.lower()
    
    if any(keyword in query_lower for keyword in ['attendance', 'present', 'absent']):
        attendance_data = erp_integration.get_attendance(user_id)
        if attendance_data:
            return f"Your attendance is {attendance_data['overall']}. Subject-wise: {', '.join([f'{sub}: {att}' for sub, att in attendance_data['subjects'].items()])}"
    
    elif any(keyword in query_lower for keyword in ['result', 'grade', 'gpa', 'cgpa', 'sgpa']):
        results_data = erp_integration.get_results(user_id)
        if results_data:
            return f"Your SGPA for {results_data['current_semester']} is {results_data['sgpa']}. CGPA is {results_data['cgpa']}. Grades: {', '.join([f'{sub}: {grade}' for sub, grade in results_data['grades'].items()])}"
    
    elif any(keyword in query_lower for keyword in ['fee', 'payment', 'paid', 'pending']):
        fee_data = erp_integration.get_fee_details(user_id)
        if fee_data:
            return f"Total fee: {fee_data['total_fee']}. Paid: {fee_data['paid']}. Pending: {fee_data['pending']}. Due date: {fee_data['due_date']}"
    
    elif any(keyword in query_lower for keyword in ['timetable', 'schedule', 'class']):
        timetable_data = erp_integration.get_timetable(user_id)
        if timetable_data:
            # Format timetable for better readability
            timetable_str = "Your timetable:\n"
            for day, classes in timetable_data.items():
                timetable_str += f"{day.capitalize()}: "
                timetable_str += ', '.join([f"{cls['time']} {cls['subject']} ({cls['room']})" for cls in classes])
                timetable_str += "\n"
            return timetable_str
    
    return None

def run_chat():
    """Run the chatbot application"""
    print("=" * 50)
    print("College AI Assistant")
    print("=" * 50)
    print("Supported languages: English (en), Hindi (hi), Marathi (mr)")
    print("Type 'exit' to quit")
    print("=" * 50)
    
    # Get user ID
    user_id = input("Enter your student ID (or leave blank for general queries): ")
    if user_id:
        if user_id not in context:
            context[user_id] = []
    
    # Get preferred language
    lang = input("Select language (en/hi/mr) [default: en]: ").lower()
    if lang not in ['en', 'hi', 'mr']:
        lang = 'en'
    voice_assistant.set_language(lang)
    
    # Get preferred AI service
    ai_service = input("Select AI service (openai/chatz) [default: openai]: ").lower()
    if ai_service not in ['openai', 'chatz']:
        ai_service = 'openai'
    
    print("\nYou can now start interacting with the chatbot.")
    print("Choose input method each time:")
    print("1. Voice")
    print("2. Text")
    print("3. Exit")
    print("=" * 50)
    
    while True:
        # Get input method
        choice = input("\nEnter your choice (1/2/3): ")
        
        if choice == '1':
            # Voice input
            message = voice_assistant.speech_to_text(lang)
            if not message:
                continue
        elif choice == '2':
            # Text input
            message = input("You: ")
        elif choice == '3':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")
            continue
        
        # Check for exit command
        if message.lower() in ['exit', 'quit', 'bye']:
            print("Goodbye!")
            break
        
        # Check for ERP-related queries
        erp_response = check_erp_query(message, user_id)
        if erp_response:
            response = erp_response
        else:
            # Get response from chatbot
            ints = predict_class(message)
            response = get_response(ints, intents, user_id)
            
            # Update context
            if user_id:
                context[user_id].append({
                    'timestamp': datetime.now().isoformat(),
                    'query': message,
                    'response': response
                })
                
                # Keep only the last 10 conversation pairs to prevent memory issues
                if len(context[user_id]) > 10:
                    context[user_id] = context[user_id][-10:]
        
        # Speak response
        voice_assistant.speak_response(response, lang)

if __name__ == "__main__":
    from datetime import datetime
    run_chat()