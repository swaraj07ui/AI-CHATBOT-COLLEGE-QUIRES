import speech_recognition as sr
import pyttsx3
from gtts import gTTS
import os
import tempfile
import platform
import pygame
import time
import sys

class RobustVoiceAssistant:
    def __init__(self):
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Adjust for ambient noise
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
        
        # Set default language
        self.language = 'en'
        
        # Initialize TTS engine with error handling
        self.tts_engine = None
        self.init_tts_engine()
        
        # Initialize pygame for audio playback
        pygame.mixer.init()
    
    def init_tts_engine(self):
        """Initialize TTS engine with error handling"""
        try:
            # Try to initialize pyttsx3
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)
            self.tts_engine.setProperty('volume', 0.9)
        except Exception as e:
            print(f"Warning: Could not initialize pyttsx3: {e}")
            print("Using gTTS for all text-to-speech operations")
            self.tts_engine = None
    
    def set_language(self, lang):
        """Set the language for speech recognition and synthesis"""
        self.language = lang
    
    def text_to_speech(self, text, lang=None):
        """Convert text to speech with multi-language support"""
        if lang is None:
            lang = self.language
        
        try:
            if lang == 'en' and self.tts_engine is not None:
                # Use pyttsx3 for English (offline)
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            else:
                # Use gTTS for other languages (online)
                self.gtts_speak(text, lang)
        except Exception as e:
            print(f"Error in text-to-speech: {e}")
            print(f"Text that should have been spoken: {text}")
            # Try gTTS as fallback
            self.gtts_speak(text, lang)
    
    def gtts_speak(self, text, lang):
        """Use gTTS for text-to-speech"""
        try:
            tts = gTTS(text=text, lang=lang, slow=False)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                temp_path = fp.name
                tts.save(temp_path)
                
                # Play the file using pygame
                pygame.mixer.music.load(temp_path)
                pygame.mixer.music.play()
                
                # Wait for playback to finish
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                
                # Clean up
                pygame.mixer.music.unload()
                os.unlink(temp_path)
        except Exception as e:
            print(f"Error with gTTS: {e}")
            print(f"Text that should have been spoken: {text}")
    
    def speech_to_text(self, lang=None):
        """Convert speech to text with multi-language support"""
        if lang is None:
            lang = self.language
        
        try:
            with self.microphone as source:
                print("Listening...")
                # Adjust for ambient noise each time
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Listen with a longer timeout and phrase time limit
                audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=15)
            
            print("Recognizing...")
            if lang == 'en':
                text = self.recognizer.recognize_google(audio)
            elif lang == 'hi':
                text = self.recognizer.recognize_google(audio, language='hi-IN')
            elif lang == 'mr':
                text = self.recognizer.recognize_google(audio, language='mr-IN')
            else:
                text = self.recognizer.recognize_google(audio)
            
            print(f"You said: {text}")
            return text
        except sr.WaitTimeoutError:
            print("Listening timed out. No speech detected.")
            return ""
        except sr.UnknownValueError:
            print("Sorry, I didn't understand that.")
            return ""
        except sr.RequestError as e:
            print(f"Error with speech recognition service; {e}")
            return ""
        except Exception as e:
            print(f"Unexpected error in speech recognition: {e}")
            return ""
    
    def speak_response(self, response, lang=None):
        """Speak the chatbot's response in the specified language"""
        if lang is None:
            lang = self.language
        
        print(f"Bot: {response}")
        self.text_to_speech(response, lang)