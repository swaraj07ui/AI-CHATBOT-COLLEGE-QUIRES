import speech_recognition as sr
import pyttsx3
from gtts import gTTS
import os
import tempfile
import platform
import time
import pygame
import atexit
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoiceAssistant:
    def __init__(self):
        self.engine = None
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.language = 'en'
        self.supported_languages = {
            'en': 'English',
            'hi': 'Hindi',
            'mr': 'Marathi'
        }
        
        # Initialize components
        self.init_tts_engine()
        self.init_speech_recognition()
        self.init_audio_player()
        
        # Register cleanup function
        atexit.register(self.cleanup)
        
        logger.info("Voice Assistant initialized successfully")
    
    def init_tts_engine(self):
        """Initialize text-to-speech engine with error handling"""
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)
            self.engine.setProperty('volume', 0.9)
            logger.info("TTS engine initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing TTS engine: {e}")
            self.engine = None
    
    def init_speech_recognition(self):
        """Initialize speech recognition with ambient noise adjustment"""
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source)
            logger.info("Speech recognition initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing speech recognition: {e}")
    
    def init_audio_player(self):
        """Initialize audio player for gTTS output"""
        try:
            pygame.mixer.init()
            logger.info("Audio player initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing audio player: {e}")
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if self.engine:
                self.engine.stop()
            pygame.mixer.quit()
            logger.info("Resources cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def set_language(self, lang):
        """Set the language for speech recognition and synthesis"""
        if lang in self.supported_languages:
            self.language = lang
            logger.info(f"Language set to {self.supported_languages[lang]}")
        else:
            logger.warning(f"Unsupported language: {lang}. Using default (English)")
            self.language = 'en'
    
    def text_to_speech(self, text, lang=None):
        """Convert text to speech with multi-language support"""
        if lang is None:
            lang = self.language
        
        try:
            if lang == 'en' and self.engine is not None:
                # Use pyttsx3 for English (offline)
                logger.info(f"Speaking in English using pyttsx3: {text}")
                self.engine.say(text)
                self.engine.runAndWait()
            else:
                # Use gTTS for other languages (online)
                logger.info(f"Speaking in {lang} using gTTS: {text}")
                self.gtts_speak(text, lang)
        except Exception as e:
            logger.error(f"Error in text-to-speech: {e}")
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
                logger.info(f"Audio saved to: {temp_path}")
                
                # Try multiple playback methods
                if self.play_audio_file(temp_path):
                    logger.info("Audio played successfully")
                else:
                    logger.warning("All playback methods failed")
                
                # Clean up
                try:
                    os.unlink(temp_path)
                    logger.info("Temporary file deleted")
                except Exception as e:
                    logger.error(f"Could not delete temporary file: {e}")
        except Exception as e:
            logger.error(f"Error with gTTS: {e}")
            logger.error(f"Text that should have been spoken: {text}")
    
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
            logger.warning(f"pygame method failed: {e}")
        
        # Method 2: playsound (if available)
        try:
            import playsound
            playsound.playsound(file_path)
            return True
        except Exception as e:
            logger.warning(f"playsound method failed: {e}")
        
        # Method 3: subprocess (OS-specific)
        try:
            system = platform.system()
            if system == 'Windows':
                import subprocess
                subprocess.call(['start', '/min', 'wmplayer', '/close', '/prefetch:8', file_path], shell=True)
                time.sleep(2)  # Wait for playback to start
                return True
            elif system == 'Darwin':  # macOS
                import subprocess
                subprocess.call(['afplay', file_path])
                return True
            else:  # Linux
                import subprocess
                subprocess.call(['mpg123', file_path])
                return True
        except Exception as e:
            logger.warning(f"subprocess method failed: {e}")
        
        return False
    
    def speech_to_text(self, lang=None):
        """Convert speech to text with multi-language support"""
        if lang is None:
            lang = self.language
        
        with self.microphone as source:
            logger.info("Listening...")
            # Adjust for ambient noise each time
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            try:
                audio = self.recognizer.listen(source, phrase_time_limit=10, timeout=5)
            except sr.WaitTimeoutError:
                logger.warning("Listening timed out")
                return ""
        
        try:
            logger.info("Recognizing...")
            if lang == 'en':
                text = self.recognizer.recognize_google(audio)
            elif lang == 'hi':
                text = self.recognizer.recognize_google(audio, language='hi-IN')
            elif lang == 'mr':
                text = self.recognizer.recognize_google(audio, language='mr-IN')
            else:
                text = self.recognizer.recognize_google(audio)
            
            logger.info(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            logger.warning("Sorry, I didn't understand that.")
            return ""
        except sr.RequestError as e:
            logger.error(f"Error with speech recognition service; {e}")
            return ""
    
    def speak_response(self, response, lang=None):
        """Speak the chatbot's response in the specified language"""
        if lang is None:
            lang = self.language
        
        print(f"Bot: {response}")
        self.text_to_speech(response, lang)