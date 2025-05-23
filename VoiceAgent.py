import streamlit as st
import speech_recognition as sr
import pyttsx3
import requests
import os
from dotenv import load_dotenv
import threading
import time
from io import BytesIO
import tempfile
try:
    from langdetect import detect
    from langdetect.lang_detect_exception import LangDetectError
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    print("Warning: langdetect not available. Using advanced pattern-based detection.")

import re
from collections import Counter
from gtts import gTTS
import pygame
import base64

# Initialize pygame mixer
pygame.mixer.init()

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Multilingual Voice Bot with Watsonx",
    page_icon="🎙️",
    layout="wide"
)

# Advanced language detection patterns and keywords
LANGUAGE_PATTERNS = {
    'hi': {
        'keywords': ['हैं', 'है', 'का', 'के', 'की', 'को', 'में', 'से', 'और', 'या', 'भी', 'नहीं', 'अपने', 'उसका', 'इसका', 'जो', 'कि', 'था', 'थी', 'होगा', 'होगी', 'कैसे', 'क्यों', 'कहाँ', 'कब'],
        'chars': range(0x0900, 0x097F),  # Devanagari
        'common_endings': ['ने', 'से', 'को', 'में', 'पर', 'ता', 'ती', 'ते'],
        'script_name': 'Devanagari'
    },
    'ta': {
        'keywords': ['அது', 'இது', 'என்', 'உன்', 'அவன்', 'அவள்', 'நான்', 'நீ', 'அவர்', 'இங்கே', 'அங்கே', 'எங்கே', 'எப்போது', 'எதற்கு', 'எப்படி', 'ஆம்', 'இல்லை', 'மற்றும்', 'அல்லது', 'ஆனால்', 'என்றால்', 'போல்', 'மேல்', 'கீழ்', 'உள்ளே', 'வெளியே'],
        'chars': range(0x0B80, 0x0BFF),  # Tamil
        'common_endings': ['ான்', 'ாள்', 'ார்', 'ிது', 'ிள்', 'ுது', 'ேன்', 'ோம்'],
        'script_name': 'Tamil'

    },
    'en': {
        'keywords': ['the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their'],
        'chars': range(0x0020, 0x007F),  # ASCII
        'common_endings': ['ing', 'ed', 'er', 'ly', 'tion', 'ness', 'ment'],
        'script_name': 'Latin'
    }
}
SUPPORTED_LANGUAGES = {
    'en': 'en-US',      # English
    'hi': 'hi-IN',      # Hindi
    'ta': 'ta-IN',      # Tamil

}

# TTS Language mapping for gTTS
TTS_LANGUAGE_MAPPING = {
    'en': 'en',
    'hi': 'hi',
    'ta': 'ta',
}

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'is_listening' not in st.session_state:
    st.session_state.is_listening = False
if 'bearer_token' not in st.session_state:
    st.session_state.bearer_token = None
if 'last_response' not in st.session_state:
    st.session_state.last_response = ""
if 'detected_language' not in st.session_state:
    st.session_state.detected_language = 'en'
if 'auto_detect' not in st.session_state:
    st.session_state.auto_detect = True

# Your existing Watsonx functions
def get_bearer_token(api_key):
    url = "https://iam.cloud.ibm.com/identity/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = f"apikey={api_key}&grant_type=urn:ibm:params:oauth:grant-type:apikey"

    response = requests.post(url, headers=headers, data=data)

    if response.status_code == 200:
        return response.json()["access_token"]
    else:
        st.error(f"Failed to retrieve access token: {response.text}")
        return None

def clean_ai_response(response_text):
    """Clean the AI response by removing template tags and unwanted text"""
    if not response_text:
        return response_text
    
    # Remove common template tags
    unwanted_patterns = [
        "assistant<|end_header_id|>",
        "<|start_header_id|>assistant<|end_header_id|>",
        "<|eot_id|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "assistant<|end_header_id|>\n\n",
        "assistant<|end_header_id|>\n",
    ]
    
    cleaned_response = response_text
    for pattern in unwanted_patterns:
        cleaned_response = cleaned_response.replace(pattern, "")
    
    # Remove leading/trailing whitespace and newlines
    cleaned_response = cleaned_response.strip()
    
    return cleaned_response

def get_watsonx_response(history, user_input, bearer_token, detected_lang='en'):
    url = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {bearer_token}"
    }

    # Add language context to the conversation
    language_context = ""
    if detected_lang != 'en':
        lang_names = {
            'hi': 'Hindi', 'ta': 'Tamil', 'te': 'Telugu', 'kn': 'Kannada',
            'ml': 'Malayalam', 'bn': 'Bengali', 'gu': 'Gujarati', 'mr': 'Marathi',
            'pa': 'Punjabi', 'or': 'Odia', 'as': 'Assamese', 'ur': 'Urdu'
        }
        lang_name = lang_names.get(detected_lang, 'regional language')
        language_context = f"The user is speaking in {lang_name}. Please respond appropriately and consider the cultural context. If needed, you can respond in English or the same language as appropriate."

    # Construct the conversation history
    conversation = ""
    if language_context:
        conversation += f"<|start_header_id|>system<|end_header_id|>\n\n{language_context}<|eot_id|>\n"
    
    conversation += "".join(
        f"<|start_header_id|>{role}<|end_header_id|>\n\n{text}<|eot_id|>\n" 
        for role, text in history
    )
    
    conversation += f"<|start_header_id|>user<|end_header_id|>\n\n{user_input}<|eot_id|>\n"

    payload = {
        "input": conversation,
        "parameters": {
            "decoding_method": "greedy",
            "max_new_tokens": 8100,
            "min_new_tokens": 0,
            "stop_sequences": [],
            "repetition_penalty": 1
        },
        "model_id": "meta-llama/llama-3-3-70b-instruct",
        "project_id": os.getenv("PROJECT_ID")
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        response_data = response.json()
        if "results" in response_data and response_data["results"]:
            raw_response = response_data["results"][0]["generated_text"]
            return clean_ai_response(raw_response)
        else:
            return "Error: 'generated_text' not found in the response."
    else:
        return f"Error: Failed to fetch response from Watsonx.ai. Status code: {response.status_code}"

def advanced_language_detection(text):
    """Advanced language detection using multiple techniques"""
    if not text or len(text.strip()) < 2:
        return 'en', 0.0
    
    text = text.lower().strip()
    scores = {}
    
    # Method 1: Script/Character range detection
    char_scores = {}
    for lang, patterns in LANGUAGE_PATTERNS.items():
        char_count = 0
        total_chars = len([c for c in text if c.isalpha()])
        
        if total_chars == 0:
            continue
            
        for char in text:
            if any(ord(char) in patterns['chars'] for _ in [1]):
                try:
                    if ord(char) in patterns['chars']:
                        char_count += 1
                except:
                    pass
        
        if total_chars > 0:
            char_scores[lang] = char_count / total_chars
    
    # Method 2: Keyword matching
    keyword_scores = {}
    words = re.findall(r'\b\w+\b', text)
    total_words = len(words)
    
    for lang, patterns in LANGUAGE_PATTERNS.items():
        keyword_matches = 0
        for word in words:
            if word in patterns['keywords']:
                keyword_matches += 1
        
        if total_words > 0:
            keyword_scores[lang] = keyword_matches / total_words
    
    # Method 3: Common endings detection
    ending_scores = {}
    for lang, patterns in LANGUAGE_PATTERNS.items():
        ending_matches = 0
        for ending in patterns['common_endings']:
            if text.endswith(ending) or any(word.endswith(ending) for word in words):
                ending_matches += 1
        
        ending_scores[lang] = ending_matches / len(patterns['common_endings'])
    
    # Method 4: Language-specific patterns
    pattern_scores = {}
    
    # Hindi/Marathi specific patterns
    hindi_patterns = [r'[हैं|है|का|के|की|को|में|से]', r'क्[या|यों|या]', r'[होगा|होगी|होंगे]']
    marathi_patterns = [r'[आहे|आहेत|चा|चे|ची|ला|मध्ये]', r'[कसे|कुठे|केव्हा]']
    
    # Tamil specific patterns
    tamil_patterns = [r'[ான்|ாள்|ார்|க்கு|வில்|டு]', r'[என்ன|எப்படி|எங்கே]']
    
    # Telugu specific patterns  
    telugu_patterns = [r'[అని|లేదు|వున్న|చేస్]', r'[ఎలా|ఎక్కడ|ఎప్పుడు]']
    
    for lang in LANGUAGE_PATTERNS.keys():
        pattern_matches = 0
        if lang == 'hi':
            pattern_matches = sum(len(re.findall(pattern, text)) for pattern in hindi_patterns)
        elif lang == 'mr':
            pattern_matches = sum(len(re.findall(pattern, text)) for pattern in marathi_patterns)
        elif lang == 'ta':
            pattern_matches = sum(len(re.findall(pattern, text)) for pattern in tamil_patterns)
        elif lang == 'te':
            pattern_matches = sum(len(re.findall(pattern, text)) for pattern in telugu_patterns)
        
        pattern_scores[lang] = pattern_matches / max(1, len(words))
    
    # Combine all scores with weights
    for lang in LANGUAGE_PATTERNS.keys():
        combined_score = (
            char_scores.get(lang, 0) * 0.4 +           # Script detection (40%)
            keyword_scores.get(lang, 0) * 0.3 +        # Keyword matching (30%)
            ending_scores.get(lang, 0) * 0.2 +         # Common endings (20%)
            pattern_scores.get(lang, 0) * 0.1          # Pattern matching (10%)
        )
        scores[lang] = combined_score
    
    # Use langdetect as additional validation if available
    if LANGDETECT_AVAILABLE:
        try:
            langdetect_result = detect(text)
            if langdetect_result in scores:
                scores[langdetect_result] *= 1.2  # Boost score by 20%
        except:
            pass
    
    # Find the language with highest score
    if scores:
        best_lang = max(scores, key=scores.get)
        confidence = scores[best_lang]
        
        # Minimum confidence threshold
        if confidence < 0.1:
            return 'en', confidence
        
        return best_lang, confidence
    
    return 'en', 0.0

def detect_language_from_text(text):
    """Enhanced language detection with confidence scoring"""
    detected_lang, confidence = advanced_language_detection(text)
    
    # Log detection results for debugging
    if hasattr(st, 'session_state'):
        st.session_state.last_detection_confidence = confidence
    
    return detected_lang

def listen_for_speech_multilingual():
    """Enhanced speech recognition with advanced language detection"""
    recognizer = sr.Recognizer()
    
    with sr.Microphone() as source:
        st.info("🎤 Listening... Speak in any supported Indian language!")
        try:
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=1)
            # Listen for speech with timeout
            audio = recognizer.listen(source, timeout=15, phrase_time_limit=15)
            
            # If auto-detect is enabled, try multiple languages with advanced detection
            if st.session_state.auto_detect:
                recognition_results = []
                
                # Priority order: commonly used languages first
                priority_languages = ['en', 'hi', 'ta', 'te', 'kn', 'ml', 'bn', 'gu', 'mr']
                other_languages = [lang for lang in SUPPORTED_LANGUAGES.keys() if lang not in priority_languages]
                languages_to_try = priority_languages + other_languages
                
                # Try recognition with each language
                for lang_code in languages_to_try:
                    try:
                        google_lang_code = SUPPORTED_LANGUAGES[lang_code]
                        text = recognizer.recognize_google(audio, language=google_lang_code)
                        
                        if text.strip():
                            # Use advanced language detection
                            detected_lang, confidence = advanced_language_detection(text)
                            
                            # Calculate total score (recognition success + language match + confidence)
                            lang_match_bonus = 1.0 if detected_lang == lang_code else 0.5
                            total_score = confidence + lang_match_bonus + (len(text.split()) * 0.1)
                            
                            recognition_results.append({
                                'text': text,
                                'recognition_lang': lang_code,
                                'detected_lang': detected_lang,
                                'confidence': confidence,
                                'total_score': total_score
                            })
                            
                    except (sr.UnknownValueError, sr.RequestError, Exception):
                        continue
                
                # Select the best result based on total score
                if recognition_results:
                    best_result = max(recognition_results, key=lambda x: x['total_score'])
                    
                    # Display detection details
                    st.session_state.last_detection_details = {
                        'recognition_lang': best_result['recognition_lang'],
                        'detected_lang': best_result['detected_lang'],
                        'confidence': best_result['confidence'],
                        'total_score': best_result['total_score'],
                        'all_results': len(recognition_results)
                    }
                    
                    final_lang = best_result['detected_lang']
                    st.session_state.detected_language = final_lang
                    
                    return best_result['text'], final_lang
                else:
                    return "Could not understand audio in any supported language", 'en'
            
            else:
                # Use manually selected language
                selected_lang = st.session_state.detected_language
                google_lang_code = SUPPORTED_LANGUAGES[selected_lang]
                text = recognizer.recognize_google(audio, language=google_lang_code)
                
                # Still run advanced detection for validation
                detected_lang, confidence = advanced_language_detection(text)
                
                st.session_state.last_detection_details = {
                    'recognition_lang': selected_lang,
                    'detected_lang': detected_lang,
                    'confidence': confidence,
                    'manual_mode': True
                }
                
                return text, detected_lang
                
        except sr.WaitTimeoutError:
            return "Timeout: No speech detected", 'en'
        except sr.UnknownValueError:
            return "Could not understand audio", 'en'
        except sr.RequestError as e:
            return f"Error with speech recognition service: {e}", 'en'

def speak_text_multilingual(text, language='en'):
    """Convert text to speech with enhanced language support using gTTS"""
    try:
        # Get the appropriate language code for gTTS
        tts_lang = TTS_LANGUAGE_MAPPING.get(language, 'en')
        
        # Create a temporary file to store the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            temp_filename = temp_file.name
        
        # Generate speech using gTTS
        tts = gTTS(text=text, lang=tts_lang, slow=False)
        tts.save(temp_filename)
        
        # Play the audio using pygame
        pygame.mixer.music.load(temp_filename)
        pygame.mixer.music.play()
        
        # Wait for the audio to finish playing
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        
        # Clean up the temporary file
        try:
            os.unlink(temp_filename)
        except:
            pass
            
    except Exception as e:
        st.error(f"Error in text-to-speech: {e}")
        # Fallback to pyttsx3 if gTTS fails
        try:
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            
            # Try to find a voice for the detected language
            for voice in voices:
                voice_lang = getattr(voice, 'languages', [])
                voice_id = voice.id.lower()
                
                if language != 'en':
                    if (any(language in lang for lang in voice_lang) or 
                        language in voice_id or 
                        TTS_LANGUAGE_MAPPING.get(language, '') in voice_id):
                        engine.setProperty('voice', voice.id)
                        break
            else:
                engine.setProperty('voice', voices[0].id)
            
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 0.9)
            
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e2:
            st.error(f"Fallback TTS also failed: {e2}")

def process_voice_input():
    """Process voice input with multilingual support"""
    if not st.session_state.bearer_token:
        st.error("Please authenticate first!")
        return
    
    # Listen for speech
    result = listen_for_speech_multilingual()
    
    if isinstance(result, tuple):
        user_text, detected_lang = result
    else:
        user_text, detected_lang = result, 'en'
    
    if user_text and not any(error in user_text for error in ["Error", "Timeout", "Could not"]):
        # Display detected language
        lang_names = {
            'en': 'English', 'hi': 'Hindi (हिंदी)', 'ta': 'Tamil (தமிழ்)', 
            'te': 'Telugu (తెలుగు)', 'kn': 'Kannada (ಕನ್ನಡ)', 'ml': 'Malayalam (മലയാളം)',
            'bn': 'Bengali (বাংলা)', 'gu': 'Gujarati (ગુજરાતી)', 'mr': 'Marathi (मराठी)',
            'pa': 'Punjabi (ਪੰਜਾਬੀ)', 'or': 'Odia (ଓଡ଼ିଆ)', 'as': 'Assamese (অসমীয়া)', 'ur': 'Urdu (اردو)'
        }
        
        detected_lang_name = lang_names.get(detected_lang, detected_lang)
        st.success(f"🗣️ **Detected Language:** {detected_lang_name}")
        
        # Show detection details if available
        if hasattr(st.session_state, 'last_detection_details'):
            details = st.session_state.last_detection_details
            with st.expander("🔍 Detection Details"):
                if details.get('manual_mode'):
                    st.info(f"**Manual Mode:** Used {lang_names.get(details['recognition_lang'], details['recognition_lang'])}")
                else:
                    st.info(f"**Recognition Language:** {lang_names.get(details['recognition_lang'], details['recognition_lang'])}")
                    st.info(f"**Detected Language:** {lang_names.get(details['detected_lang'], details['detected_lang'])}")
                    st.info(f"**Confidence Score:** {details['confidence']:.2f}")
                    st.info(f"**Total Score:** {details['total_score']:.2f}")
                    st.info(f"**Languages Tried:** {details['all_results']}")
        
        st.success(f"📝 **You said:** {user_text}")
        
        # Add user input to conversation history
        st.session_state.conversation_history.append(("user", user_text))
        
        # Get AI response with language context
        with st.spinner("Getting AI response..."):
            ai_response = get_watsonx_response(
                st.session_state.conversation_history, 
                user_text, 
                st.session_state.bearer_token,
                detected_lang
            )
        
        if ai_response and not ai_response.startswith("Error"):
            # Add AI response to conversation history
            st.session_state.conversation_history.append(("assistant", ai_response))
            st.session_state.last_response = ai_response
            
            st.success(f"🤖 **AI Response:** {ai_response}")
            
            # Speak the response in appropriate language
            with st.spinner("Speaking response..."):
                speak_text_multilingual(ai_response, detected_lang)
        else:
            st.error(f"AI Error: {ai_response}")
    else:
        st.error(f"Speech Recognition Error: {user_text}")

# Main UI
st.title("🎙️ Multilingual Voice Bot with Watsonx LLM")
st.markdown("### Supports English + Indian Regional Languages")
st.markdown("---")

# Auto-authentication on app start
if not st.session_state.bearer_token:
    api_key = os.getenv("API_KEY")
    project_id = os.getenv("PROJECT_ID")
    
    if api_key and project_id:
        with st.spinner("Authenticating with Watsonx..."):
            token = get_bearer_token(api_key)
            if token:
                st.session_state.bearer_token = token
                st.success("✅ Authentication successful!")
            else:
                st.error("❌ Authentication failed! Please check your API_KEY in .env file")
    else:
        st.error("❌ Missing API_KEY or PROJECT_ID in environment variables. Please check your .env file.")

st.markdown("---")

# Language settings section
st.header("🌐 Language Settings")
col1, col2 = st.columns(2)

with col1:
    auto_detect = st.checkbox("🔍 Auto-detect language", value=st.session_state.auto_detect)
    st.session_state.auto_detect = auto_detect

with col2:
    if not auto_detect:
        lang_options = {
            'English': 'en', 'Hindi (हिंदी)': 'hi', 'Tamil (தமிழ்)': 'ta',
            'Telugu (తెలుగు)': 'te', 'Kannada (ಕನ್ನಡ)': 'kn', 'Malayalam (മലയാളം)': 'ml',
            'Bengali (বাংলা)': 'bn', 'Gujarati (ગુજરાતી)': 'gu', 'Marathi (मराठी)': 'mr',
            'Punjabi (ਪੰਜਾਬੀ)': 'pa', 'Odia (ଓଡ଼ିଆ)': 'or', 'Assamese (অসমীয়া)': 'as', 'Urdu (اردو)': 'ur'
        }
        
        selected_lang_name = st.selectbox(
            "Select Language:",
            options=list(lang_options.keys()),
            index=0
        )
        st.session_state.detected_language = lang_options[selected_lang_name]

# Display current language setting
if st.session_state.auto_detect:
    st.info("🔍 **Mode:** Auto-detect (will try to identify the language you speak)")
else:
    current_lang = next(name for name, code in lang_options.items() if code == st.session_state.detected_language)
    st.info(f"🗣️ **Selected Language:** {current_lang}")

st.markdown("---")

# Voice interaction section
st.header("🎤 Voice Interaction")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🎙️ Start Voice Chat", disabled=not st.session_state.bearer_token):
        process_voice_input()

with col2:
    if st.button("🔊 Repeat Last Response", disabled=not st.session_state.last_response):
        with st.spinner("Speaking..."):
            speak_text_multilingual(st.session_state.last_response, st.session_state.detected_language)

with col3:
    if st.button("🗑️ Clear Conversation"):
        st.session_state.conversation_history = []
        st.session_state.last_response = ""
        st.success("Conversation cleared!")

st.markdown("---")

# # Text input as backup
# st.header("💬 Text Input (Backup)")
# text_input = st.text_input("Type your message here (any language):")
# if st.button("Send Text", disabled=not st.session_state.bearer_token):
#     if text_input:
#         # Detect language from text input
#         detected_lang = detect_language_from_text(text_input)
        
#         # Add user input to conversation history
#         st.session_state.conversation_history.append(("user", text_input))
        
#         # Get AI response
#         with st.spinner("Getting AI response..."):
#             ai_response = get_watsonx_response(
#                 st.session_state.conversation_history, 
#                 text_input, 
#                 st.session_state.bearer_token,
#                 detected_lang
#             )
        
#         if ai_response and not ai_response.startswith("Error"):
#             # Add AI response to conversation history
#             st.session_state.conversation_history.append(("assistant", ai_response))
#             st.session_state.last_response = ai_response
            
#             st.success(f"AI Response: {ai_response}")
            
#             # Option to speak the response
#             if st.button("🔊 Speak Response"):
#                 speak_text_multilingual(ai_response, detected_lang)
#         else:
#             st.error(f"AI Error: {ai_response}")

# st.markdown("---")

# Conversation history display
st.header("📝 Conversation History")
if st.session_state.conversation_history:
    for i, (role, text) in enumerate(st.session_state.conversation_history):
        if role == "user":
            st.markdown(f"**👤 You:** {text}")
        else:
            st.markdown(f"**🤖 Assistant:** {text}")
            # Add individual speak button for each response
            if st.button(f"🔊 Speak", key=f"speak_{i}"):
                speak_text_multilingual(text, st.session_state.detected_language)
else:
    st.info("No conversation yet. Start by clicking 'Start Voice Chat' or typing a message.")

# # Settings section
# st.markdown("---")
# st.header("⚙️ Settings & Information")

# with st.expander("🌍 Supported Languages"):
#     st.markdown("""
#     **Currently Supported Languages:**
#     - 🇺🇸 **English** (en-US)
#     - 🇮🇳 **Hindi** (hi-IN) - हिंदी
#     - 🇮🇳 **Tamil** (ta-IN) - தமிழ்
#     - 🇮🇳 **Telugu** (te-IN) - తెలుగు
#     - 🇮🇳 **Kannada** (kn-IN) - ಕನ್ನಡ
#     - 🇮🇳 **Malayalam** (ml-IN) - മലയാളം
#     - 🇮🇳 **Bengali** (bn-IN) - বাংলা
#     - 🇮🇳 **Gujarati** (gu-IN) - ગુજરાતી
#     - 🇮🇳 **Marathi** (mr-IN) - मराठी
#     - 🇮🇳 **Punjabi** (pa-IN) - ਪੰਜਾਬੀ
#     - 🇮🇳 **Odia** (or-IN) - ଓଡ଼ିଆ
#     - 🇮🇳 **Assamese** (as-IN) - অসমীয়া
#     - 🇮🇳 **Urdu** (ur-IN) - اردو
#     """)





# # Add voice debugging to the settings section
# with st.expander("🔊 Voice Settings"):
#     st.markdown("""
#     **Text-to-Speech Engine:**
#     - Primary: Google Text-to-Speech (gTTS)
#     - Fallback: pyttsx3
    
#     **Supported Languages:**
#     - 🇺🇸 English (en)
#     - 🇮🇳 Hindi (hi)
#     - 🇮🇳 Tamil (ta)
#     - 🇮🇳 Telugu (te)
#     - 🇮🇳 Kannada (kn)
#     - 🇮🇳 Malayalam (ml)
#     - 🇮🇳 Bengali (bn)
#     - 🇮🇳 Gujarati (gu)
#     - 🇮🇳 Marathi (mr)
#     - 🇮🇳 Punjabi (pa)
#     - 🇮🇳 Odia (or)
#     - 🇮🇳 Assamese (as)
#     - 🇮🇳 Urdu (ur)
#     """)
    
#     if st.button("Show Available System Voices"):
#         voices = get_available_voices()
#         st.json(voices)

# Status indicators
st.sidebar.header("📊 Status")
st.sidebar.success("✅ Ready" if st.session_state.bearer_token else "❌ Not Authenticated")
st.sidebar.info(f"💬 Messages: {len(st.session_state.conversation_history)}")

# Current language status
if st.session_state.detected_language:
    lang_names = {
        'en': 'English', 'hi': 'Hindi', 'ta': 'Tamil', 'te': 'Telugu',
        'kn': 'Kannada', 'ml': 'Malayalam', 'bn': 'Bengali', 'gu': 'Gujarati',
        'mr': 'Marathi', 'pa': 'Punjabi', 'or': 'Odia', 'as': 'Assamese', 'ur': 'Urdu'
    }
    current_lang = lang_names.get(st.session_state.detected_language, 'Unknown')
    st.sidebar.info(f"🗣️ Language: {current_lang}")

# Help section


# Add a function to get available voices
def get_available_voices():
    """Get list of available voices for debugging"""
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        voice_info = []
        for voice in voices:
            voice_info.append({
                'id': voice.id,
                'name': voice.name,
                'languages': getattr(voice, 'languages', []),
                'gender': getattr(voice, 'gender', 'unknown')
            })
        return voice_info
    except Exception as e:
        return [{'error': str(e)}]