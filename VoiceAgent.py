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
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
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

# Initialize pygame mixer with error handling
try:
    pygame.mixer.init()
    PYGAME_AVAILABLE = True
except Exception as e:
    PYGAME_AVAILABLE = False
    print("Warning: pygame audio not available. Using alternative audio playback.")

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
if 'continuous_mode' not in st.session_state:
    st.session_state.continuous_mode = False

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
        "**",
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
            'hi': 'Hindi', 'ta': 'Tamil'
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
    
    
    # Tamil specific patterns
    tamil_patterns = [r'[ான்|ாள்|ார்|க்கு|வில்|டு]', r'[என்ன|எப்படி|எங்கே]']
    
    # Telugu specific patterns  
    
    
    for lang in LANGUAGE_PATTERNS.keys():
        pattern_matches = 0
        if lang == 'hi':
            pattern_matches = sum(len(re.findall(pattern, text)) for pattern in hindi_patterns)
        elif lang == 'ta':
            pattern_matches = sum(len(re.findall(pattern, text)) for pattern in tamil_patterns)
        
        
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
            # Adjust for ambient noise with longer duration
            recognizer.adjust_for_ambient_noise(source, duration=1)
            
            # Configure speech recognition parameters for more patient listening
            recognizer.dynamic_energy_threshold = True
            recognizer.energy_threshold = 300  # Keep threshold for noise filtering
            recognizer.pause_threshold = 1.5   # Increased pause threshold to wait longer between phrases
            recognizer.phrase_threshold = 0.3  # Keep phrase threshold the same
            recognizer.non_speaking_duration = 1.0  # Increased non-speaking duration
            
            # Listen for speech with longer timeouts
            audio = recognizer.listen(
                source,
                timeout=45,           # Increased timeout to 45 seconds
                phrase_time_limit=45  # Increased phrase time limit to 45 seconds
            )
            
            # If auto-detect is enabled, try multiple languages with advanced detection
            if st.session_state.auto_detect:
                recognition_results = []
                
                # Priority order: commonly used languages first
                priority_languages = ['en', 'hi', 'ta']
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
                            
                    except (sr.UnknownValueError, sr.RequestError, Exception) as e:
                        st.warning(f"Failed to recognize speech in {lang_code}: {str(e)}")
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
                    st.error("Could not understand audio in any supported language. Please try speaking again.")
                    return "Could not understand audio in any supported language", 'en'
            
            else:
                # Use manually selected language
                selected_lang = st.session_state.detected_language
                google_lang_code = SUPPORTED_LANGUAGES[selected_lang]
                try:
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
                except sr.UnknownValueError:
                    st.error(f"Could not understand audio in {selected_lang}. Please try speaking again.")
                    return "Could not understand audio", 'en'
                except sr.RequestError as e:
                    st.error(f"Error with speech recognition service: {e}")
                    return f"Error with speech recognition service: {e}", 'en'
                
        except sr.WaitTimeoutError:
            st.error("Timeout: No speech detected. Please try speaking again.")
            return "Timeout: No speech detected", 'en'
        except sr.UnknownValueError:
            st.error("Could not understand audio. Please try speaking again.")
            return "Could not understand audio", 'en'
        except sr.RequestError as e:
            st.error(f"Error with speech recognition service: {e}")
            return f"Error with speech recognition service: {e}", 'en'
        except Exception as e:
            st.error(f"Error during speech recognition: {str(e)}")
            return f"Error during speech recognition: {str(e)}", 'en'

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
        
        if PYGAME_AVAILABLE:
            # Play the audio using pygame
            pygame.mixer.music.load(temp_filename)
            pygame.mixer.music.play()
            
            # Wait for the audio to finish playing
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        else:
            # Alternative: Create an audio player using Streamlit's audio component
            with open(temp_filename, 'rb') as audio_file:
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format='audio/mp3')
            
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
    
    while st.session_state.continuous_mode:
        try:
            # Listen for speech
            result = listen_for_speech_multilingual()
            
            if isinstance(result, tuple):
                user_text, detected_lang = result
            else:
                user_text, detected_lang = result, 'en'
            
            # Check if the text is too short (might indicate cutoff)
            if len(user_text.split()) < 3 and not any(error in user_text for error in ["Error", "Timeout", "Could not"]):
                st.warning("Speech might have been cut off. Please try speaking again.")
                continue
            
            if user_text and not any(error in user_text for error in ["Error", "Timeout", "Could not"]):
                # Display detected language
                lang_names = {
                    'en': 'English', 'hi': 'Hindi (हिंदी)', 'ta': 'Tamil (தமிழ்)'
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
                # Don't break on error, continue listening
                continue
                
        except Exception as e:
            st.error(f"Error in voice processing: {str(e)}")
            # Don't break on error, continue listening
            continue

def get_conversation_summary(conversation_history):
    """Generate a summary of the conversation using Watsonx"""
    if not conversation_history:
        return "No conversation to summarize."
    
    # Format conversation for summary
    conversation_text = "\n".join([f"{role}: {text}" for role, text in conversation_history])
    
    # Create summary prompt
    summary_prompt = f"""Please provide a concise summary of the following conversation:

{conversation_text}

Summary:"""
    
    # Get summary from Watsonx
    try:
        summary = get_watsonx_response(
            [],  # Empty history for summary
            summary_prompt,
            st.session_state.bearer_token,
            'en'  # Always use English for summaries
        )
        return summary
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def send_to_slack(summary, webhook_url):
    """Send conversation summary to Slack channel"""
    try:
        # Format the message for Slack
        message = {
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "🎙️ Voice Bot Conversation Summary",
                        "emoji": True
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Summary:*\n{summary}"
                    }
                },
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                        }
                    ]
                }
            ]
        }
        
        # Send to Slack
        response = requests.post(webhook_url, json=message)
        response.raise_for_status()
        return True
    except Exception as e:
        st.error(f"Error sending to Slack: {str(e)}")
        return False

def send_summary_email(summary, recipient_email):
    """Send conversation summary via email and Slack"""
    try:
        # Get email configuration from environment variables
        sender_email = os.getenv("EMAIL_SENDER")
        email_password = os.getenv("EMAIL_PASSWORD")
        smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL")

        if not all([sender_email, email_password]):
            return "Email configuration missing. Please set EMAIL_SENDER and EMAIL_PASSWORD in .env file."

        # Create message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = f"Voice Bot Conversation Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        # Add summary to email body
        body = f"""
        <html>
            <body>
                <h2>Voice Bot Conversation Summary</h2>
                <p>Here is the summary of your recent conversation:</p>
                <div style="background-color: #f5f5f5; padding: 15px; border-radius: 5px;">
                    {summary}
                </div>
                <p>This summary was generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </body>
        </html>
        """
        msg.attach(MIMEText(body, 'html'))

        # Send email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, email_password)
            server.send_message(msg)

        # Send to Slack if webhook URL is configured
        if slack_webhook_url:
            slack_success = send_to_slack(summary, slack_webhook_url)
            if slack_success:
                return "Summary sent successfully to email and Slack!"
            else:
                return "Summary sent to email but failed to send to Slack."
        
        return "Summary sent successfully to email!"
    except Exception as e:
        return f"Error sending summary: {str(e)}"

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
            'English': 'en', 'Hindi (हिंदी)': 'hi', 'Tamil (தமிழ்)': 'ta'
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
    if st.button("🎙️ Start Continuous Voice Chat", disabled=not st.session_state.bearer_token):
        st.session_state.continuous_mode = True
        process_voice_input()

with col2:
    if st.button("⏹️ Stop Voice Chat"):
        st.session_state.continuous_mode = False
        st.success("Voice chat stopped!")

with col3:
    if st.button("🗑️ Clear Conversation"):
        st.session_state.conversation_history = []
        st.session_state.last_response = ""
        st.success("Conversation cleared!")

st.markdown("---")

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

# Add this after the conversation history display section
st.markdown("---")
st.header("📊 Conversation Summary")

# Add email input field
email_address = st.text_input("Enter email address to receive summary:", key="summary_email")

# Add a button to generate and send summary
col1, col2 = st.columns(2)
with col1:
    if st.button("Generate Summary"):
        with st.spinner("Generating conversation summary..."):
            summary = get_conversation_summary(st.session_state.conversation_history)
            st.markdown("### Summary")
            st.markdown(summary)
            
            # Store summary in session state
            st.session_state.last_summary = summary

with col2:
    if st.button("Send Summary via Email", disabled=not email_address):
        if not st.session_state.get('last_summary'):
            st.warning("Please generate a summary first!")
        else:
            with st.spinner("Sending summary via email..."):
                result = send_summary_email(st.session_state.last_summary, email_address)
                if "successfully" in result:
                    st.success(result)
                else:
                    st.error(result)

# Add automatic summary every 5 messages
if len(st.session_state.conversation_history) > 0 and len(st.session_state.conversation_history) % 5 == 0:
    with st.spinner("Generating periodic summary..."):
        summary = get_conversation_summary(st.session_state.conversation_history)
        st.markdown("### Periodic Summary")
        st.markdown(summary)
        
        # Store summary in session state
        st.session_state.last_summary = summary
        
        # If email is provided, offer to send the periodic summary
        if email_address:
            if st.button("Send Periodic Summary via Email"):
                with st.spinner("Sending periodic summary via email..."):
                    result = send_summary_email(summary, email_address)
                    if "successfully" in result:
                        st.success(result)
                    else:
                        st.error(result)

# Status indicators
st.sidebar.header("📊 Status")
st.sidebar.success("✅ Ready" if st.session_state.bearer_token else "❌ Not Authenticated")
st.sidebar.info(f"💬 Messages: {len(st.session_state.conversation_history)}")

# Current language status
if st.session_state.detected_language:
    lang_names = {
        'en': 'English', 'hi': 'Hindi', 'ta': 'Tamil'
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
