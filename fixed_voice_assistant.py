# Create fixed_voice_assistant.py
import speech_recognition as sr
import pyttsx3
from gtts import gTTS
import os
import tempfile
import platform
import time
import pygame

class FixedVoiceAssistant:
    def __init__(self):
        # Initialize text-to-speech engine with error handling
        self.engine = None
        self.init_tts_engine()
        
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Adjust for ambient noise
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
        
        # Set default language
        self.language = 'en'
        
        # Initialize pygame mixer for audio playback
        pygame.mixer.init()
    
    def init_tts_engine(self):
        """Initialize TTS engine with error handling"""
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)  # Speed of speech
            self.engine.setProperty('volume', 0.9)  # Volume level
            print("TTS engine initialized successfully")
        except Exception as e:
            print(f"Error initializing TTS engine: {e}")
            self.engine = None
    
    def set_language(self, lang):
        """Set the language for speech recognition and synthesis"""
        self.language = lang
    
    def text_to_speech(self, text, lang=None):
        """Convert text to speech with multi-language support"""
        if lang is None:
            lang = self.language
        
        try:
            if lang == 'en' and self.engine is not None:
                # Use pyttsx3 for English (offline)
                print(f"Speaking in English using pyttsx3: {text}")
                self.engine.say(text)
                self.engine.runAndWait()
            else:
                # Use gTTS for other languages (online)
                print(f"Speaking in {lang} using gTTS: {text}")
                self.gtts_speak(text, lang)
        except Exception as e:
            print(f"Error in text-to-speech: {e}")
            # Try gTTS as fallback even for English
            self.gtts_speak(text, lang)
    
    def gtts_speak(self, text, lang):
        """Use gTTS for text-to-speech with multiple playback methods"""
        try:
            # Create TTS
            tts = gTTS(text=text, lang=lang, slow=False)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                temp_path = fp.name
                tts.save(temp_path)
                print(f"Audio saved to: {temp_path}")
                
                # Try multiple playback methods
                if self.play_audio_file(temp_path):
                    print("Audio played successfully")
                else:
                    print("All playback methods failed")
                
                # Clean up
                try:
                    os.unlink(temp_path)
                    print("Temporary file deleted")
                except:
                    print("Could not delete temporary file")
        except Exception as e:
            print(f"Error with gTTS: {e}")
            print(f"Text that should have been spoken: {text}")
    
    def play_audio_file(self, file_path):
        """Try multiple methods to play an audio file"""
        # Method 1: pygame (most reliable)
        try:
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            
            pygame.mixer.music.unload()
            return True
        except Exception as e:
            print(f"pygame method failed: {e}")
        
        # Method 2: playsound (if available)
        try:
            import playsound
            playsound.playsound(file_path)
            return True
        except Exception as e:
            print(f"playsound method failed: {e}")
        
        # Method 3: subprocess (OS-specific)
        try:
            system = platform.system()
            if system == 'Windows':
                subprocess.call(['start', '/min', 'wmplayer', '/close', '/prefetch:8', file_path], shell=True)
                time.sleep(2)  # Wait for playback to start
                return True
            elif system == 'Darwin':  # macOS
                subprocess.call(['afplay', file_path])
                return True
            else:  # Linux
                subprocess.call(['mpg123', file_path])
                return True
        except Exception as e:
            print(f"subprocess method failed: {e}")
        
        return False
    
    def speech_to_text(self, lang=None):
        """Convert speech to text with multi-language support"""
        if lang is None:
            lang = self.language
        
        with self.microphone as source:
            print("Listening...")
            # Adjust for ambient noise each time
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            try:
                audio = self.recognizer.listen(source, phrase_time_limit=10, timeout=5)
            except sr.WaitTimeoutError:
                print("Listening timed out")
                return ""
        
        try:
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
        except sr.UnknownValueError:
            print("Sorry, I didn't understand that.")
            return ""
        except sr.RequestError as e:
            print(f"Error with speech recognition service; {e}")
            return ""
    
    def speak_response(self, response, lang=None):
        """Speak the chatbot's response in the specified language"""
        if lang is None:
            lang = self.language
        
        print(f"Bot: {response}")
        self.text_to_speech(response, lang)