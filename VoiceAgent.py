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
    LANGDETECT_AVAILABLE = True
except ImportError:
    print("Warning: langdetect not available. Using fallback language detection.")
    LANGDETECT_AVAILABLE = False
import re
from collections import Counter
from gtts import gTTS
import pygame
import base64
from collections import defaultdict
import math
import numpy as np

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

bot_name = "ava"

# Script ranges for Indian languages
SCRIPT_RANGES = {
    'en': (0x0041, 0x007A),  # Latin (English)
    'hi': (0x0900, 0x097F),  # Devanagari
    'ta': (0x0B80, 0x0BFF),  # Tamil
}

# Common words and patterns for each language
LANGUAGE_PATTERNS = {
    'hi': {
        'words': [
            # Common verbs
            'हैं', 'है', 'था', 'थी', 'थे', 'होगा', 'होगी', 'होंगे', 'करना', 'करेंगे', 'करूंगा', 'करेंगी',
            'आना', 'जाना', 'खाना', 'पीना', 'सोना', 'उठना', 'बैठना', 'देखना', 'सुनना', 'बोलना',
            'पढ़ना', 'लिखना', 'चलना', 'दौड़ना', 'हंसना', 'रोना', 'गाना', 'नाचना', 'खेलना', 'काम करना',
            # Common pronouns
            'मैं', 'हम', 'तुम', 'आप', 'वह', 'यह', 'वे', 'ये', 'मुझे', 'हमें', 'तुम्हें', 'आपको',
            'मेरा', 'मेरी', 'मेरे', 'हमारा', 'हमारी', 'हमारे', 'तुम्हारा', 'तुम्हारी', 'तुम्हारे',
            'आपका', 'आपकी', 'आपके', 'उसका', 'उसकी', 'उसके', 'इसका', 'इसकी', 'इसके',
            # Common postpositions
            'का', 'के', 'की', 'को', 'में', 'से', 'पर', 'तक', 'द्वारा', 'साथ', 'बिना', 'लिए',
            'ऊपर', 'नीचे', 'आगे', 'पीछे', 'बीच', 'पास', 'दूर', 'अंदर', 'बाहर', 'सामने',
            # Common conjunctions
            'और', 'या', 'लेकिन', 'क्योंकि', 'अगर', 'तो', 'मगर', 'परंतु', 'इसलिए', 'कि',
            'जब', 'जैसे', 'जितना', 'जहां', 'तब', 'वैसे', 'उतना', 'वहां', 'फिर', 'अभी',
            # Common question words
            'क्या', 'कौन', 'कहाँ', 'कब', 'कैसे', 'क्यों', 'कितना', 'कौन सा', 'किसका', 'किससे',
            # Common adjectives
            'अच्छा', 'बुरा', 'बड़ा', 'छोटा', 'नया', 'पुराना', 'ठंडा', 'गरम', 'सुंदर', 'बदसूरत',
            'लंबा', 'छोटा', 'मोटा', 'पतला', 'तेज़', 'धीमा', 'ऊंचा', 'नीचा', 'रंगीन', 'सफ़ेद',
            'काला', 'लाल', 'हरा', 'नीला', 'पीला', 'गुलाबी', 'भूरा', 'धूसर',
            # Common adverbs
            'बहुत', 'थोड़ा', 'ज्यादा', 'कम', 'अभी', 'फिर', 'भी', 'नहीं', 'हां', 'जी',
            'कल', 'आज', 'कभी', 'हमेशा', 'जल्दी', 'देर', 'धीरे', 'तेज़ी', 'यहाँ', 'वहाँ',
            # Numbers
            'एक', 'दो', 'तीन', 'चार', 'पांच', 'छह', 'सात', 'आठ', 'नौ', 'दस',
            'ग्यारह', 'बारह', 'तेरह', 'चौदह', 'पंद्रह', 'सोलह', 'सत्रह', 'अठारह', 'उन्नीस', 'बीस',
            # Time expressions
            'सुबह', 'दोपहर', 'शाम', 'रात', 'दिन', 'हफ्ता', 'महीना', 'साल', 'समय', 'घंटा',
            # Common nouns
            'घर', 'परिवार', 'माता', 'पिता', 'भाई', 'बहन', 'बच्चा', 'आदमी', 'औरत', 'लड़का', 'लड़की',
            'पानी', 'खाना', 'रोटी', 'चावल', 'दूध', 'चाय', 'कॉफी', 'फल', 'सब्जी'
        ],
        'patterns': [
            # Verb patterns - Present tense
            r'[ता|ती|ते]\s+[हैं|है|हूं]',
            r'[रहा|रही|रहे]\s+[हैं|है|हूं]',
            r'[चुका|चुकी|चुके]\s+[हैं|है|हूं]',
            # Verb patterns - Past tense
            r'[आ|ई|ए]\s+[था|थी|थे]',
            r'[करके|आकर|जाकर|देखकर]',
            # Verb patterns - Future tense
            r'[गा|गी|गे]',
            r'[ऊंगा|ऊंगी|ेंगे|ेंगी]',
            # Postposition patterns
            r'[का|के|की|को|में|से|पर|तक]',
            r'[द्वारा|साथ|बिना|लिए]',
            r'[ऊपर|नीचे|आगे|पीछे|बीच|पास|दूर|अंदर|बाहर]',
            # Question patterns
            r'क्[या|यों|या]',
            r'[कहाँ|कब|कैसे|क्यों|कितना|कौन]',
            # Word ending patterns
            r'[ने|से|को|में|पर|ता|ती|ते]$',
            r'[गा|गी|गे|ना|नी|ने]$',
            r'[वाला|वाली|वाले]$',
            r'[इया|ियां|इयों]$',
            # Honorific patterns
            r'[जी|साहब|महोदय|श्रीमान|श्रीमती]',
            # Conjunctive particles
            r'[भी|तो|ही|तक|सिर्फ|केवल]',
            # Common Hindi word patterns
            r'[हिंदी|भारत|देश|समय|दिन|रात|सुबह|शाम]',
            # Compound verb patterns
            r'[दे|ले|आ|जा]\s+[दिया|लिया|आया|गया]',
            # Negative patterns
            r'न[हीं|ही]',
            r'मत',
            # Conditional patterns
            r'[अगर|यदि].*तो',
            # Relative-correlative patterns
            r'[जो|जिस|जहां].*[वो|उस|वहां]'
        ]
    },
    'ta': {
        'words': [
            # Common pronouns
            'நான்', 'நாங்கள்', 'நாம்', 'நீ', 'நீங்கள்', 'அவன்', 'அவள்', 'அவர்', 'அவர்கள்', 
            'இது', 'அது', 'இவை', 'அவை', 'எது', 'யார்', 'எவர்',
            'என்', 'எங்கள்', 'எனது', 'எங்களது', 'உன்', 'உங்கள்', 'உனது', 'உங்களது',
            'அவன்', 'அவனது', 'அவள்', 'அவளது', 'அவர்', 'அவரது', 'அவர்கள்', 'அவர்களது',
            # Common verbs
            'உள்ளது', 'இல்லை', 'வருகிறேன்', 'போகிறேன்', 'செய்கிறேன்', 'பார்க்கிறேன்', 'கேட்கிறேன்',
            'வந்தேன்', 'போனேன்', 'செய்தேன்', 'பார்த்தேன்', 'கேட்டேன்', 'சாப்பிட்டேன்', 'குடித்தேன்',
            'வருவேன்', 'போவேன்', 'செய்வேன்', 'பார்ப்பேன்', 'கேட்பேன்', 'சாப்பிடுவேன்', 'குடிப்பேன்',
            'படிக்கிறேன்', 'எழுதுகிறேன்', 'நடக்கிறேன்', 'ஓடுகிறேன்', 'சிரிக்கிறேன்', 'அழுகிறேன்',
            'பாடுகிறேன்', 'ஆடுகிறேன்', 'விளையாடுகிறேன்', 'வேலை செய்கிறேன்',
            # Common postpositions
            'இல்', 'இடம்', 'வரை', 'மூலம்', 'ஆக', 'ஆல்', 'உடன்', 'இல்லாமல்', 'போல்',
            'மேல்', 'கீழ்', 'முன்', 'பின்', 'நடுவில்', 'அருகில்', 'தொலைவில்', 'உள்ளே', 'வெளியே',
            # Common conjunctions
            'மற்றும்', 'அல்லது', 'ஆனால்', 'என்றால்', 'ஏனெனில்', 'ஆகையால்', 'எனவே',
            'எப்போது', 'போல்', 'எவ்வளவு', 'எங்கே', 'எப்படி', 'இன்னும்', 'கூட',
            # Common question words
            'என்ன', 'எப்படி', 'எங்கே', 'எப்போது', 'ஏன்', 'எத்தனை', 'எந்த', 'யார்',
            'எது', 'எவர்', 'எவை', 'எதை', 'யாரை', 'எங்கிருந்து', 'எங்கு',
            # Common adjectives
            'நல்ல', 'கெட்ட', 'பெரிய', 'சிறிய', 'புதிய', 'பழைய', 'குளிர்ந்த', 'சூடான',
            'நீண்ட', 'குறுகிய', 'தடிமான', 'மெல்லிய', 'வேகமான', 'மெதுவான', 'உயர்ந்த', 'தாழ்ந்த',
            'அழகான', 'அசிங்கமான', 'வெள்ளை', 'கருப்பு', 'சிவப்பு', 'பச்சை', 'நீலம்', 'மஞ்சள்',
            'இளஞ்சிவப்பு', 'பழுப்பு', 'சாம்பல்',
            # Common adverbs
            'மிகவும்', 'கொஞ்சம்', 'அதிகம்', 'குறைவாக', 'இப்போது', 'மீண்டும்', 'உம்', 'இல்லை',
            'நேற்று', 'இன்று', 'நாளை', 'எப்போதும்', 'எப்போதாவது', 'சீக்கிரம்', 'தாமதம்', 'மெதுவாக',
            # Numbers
            'ஒன்று', 'இரண்டு', 'மூன்று', 'நான்கு', 'ஐந்து', 'ஆறு', 'ஏழு', 'எட்டு', 'ஒன்பது', 'பத்து',
            'பதினொன்று', 'பனிரெண்டு', 'பதிமூன்று', 'பதினான்கு', 'பதினைந்து', 'பதினாறு', 'பதினேழு',
            'பதினெட்டு', 'பத்தொன்பது', 'இருபது',
            # Time expressions
            'காலை', 'மதியம்', 'மாலை', 'இரவு', 'நாள்', 'வாரம்', 'மாதம்', 'வருடம்', 'நேரம்', 'மணி',
            # Common nouns
            'வீடு', 'குடும்பம்', 'அம்மா', 'அப்பா', 'அண்ணன்', 'தம்பி', 'அக்காள்', 'தங்கை',
            'குழந்தை', 'ஆண்', 'பெண்', 'பையன்', 'பெண்',
            'தண்ணீர்', 'சாப்பாடு', 'சோறு', 'ரொட்டி', 'பால்', 'டீ', 'காபி', 'பழம்', 'காய்கறி'
        ],
        'patterns': [
            # Verb patterns - Present tense
            r'[கிற|ற][ேன்|ாய்|ான்|ாள்|ார்|ோம்|ீர்கள்|ார்கள்]',
            r'[ன்|ள்|ர்|ம்|ங்கள்]$',
            # Verb patterns - Past tense
            r'[ந்த|ட்ட|த்த|ற்ற][ேன்|ாய்|ான்|ாள்|ார்|ோம்|ீர்கள்|ார்கள்]',
            r'[த்|ட்|ன்|ர்][த|ட]',
            # Verb patterns - Future tense
            r'[வ|ப்ப|ட்][ேன்|ாய்|ான்|ாள்|ார்|ோம்|ீர்கள்|ார்கள்]',
            # Question patterns
            r'[என்ன|எப்படி|எங்கே|எப்போது|ஏன்|எத்தனை|யார்]',
            r'[எது|எந்த|எவர்|எவை]',
            # Word ending patterns
            r'[ன்|ள்|ர்|து|ும்|ேன்|ோம்|ால்|உக்கு|இல்|அது]$',
            r'[கிற|ந்த|வ|க்கு|வில்|டு|ஆல்|உடன்]',
            # Postposition patterns
            r'[இல்|வில்|ஆல்|உடன்|மூலம்|வரை|பிறகு]',
            r'[மேல்|கீழ்|முன்|பின்|அருகில்|நடுவில்]',
            # Case marker patterns
            r'[ஐ|அ|உக்கு|ஆல்|இல்|ிடம்|ோடு]$',
            # Honorific patterns
            r'[அவர்கள்|தாங்கள்|இவர்கள்]',
            # Plural patterns
            r'[கள்|ங்கள்]$',
            # Compound verb patterns
            r'[கொண்டு|விட்டு|போட்டு]\s+[வர|போ|கொள்|தர|கொடு]',
            # Common Tamil word patterns
            r'[தமிழ்|இந்தியா|நாடு|காலம்|நாள்|இரவு|காலை|மாலை]',
            # Number patterns with Tamil numerals
            r'[௧|௨|௩|௪|௫|௬|௭|௮|௯|௦]',
            # Conjunctive particles
            r'[உம்|ேனும்|ாவது|கூட|மட்டும்|தான்]',
            # Relative patterns
            r'[எந்த|எவ].*[அந்த|அவ]',
            # Negative patterns
            r'[இல்லை|மாட்|ாமல்|வேண்டாம்]',
            # Special Telugu patterns
            r'[ஆ|ஈ|ஊ|ஏ|ஐ|ஓ|ஔ]',
            r'[க்|ங்|ச்|ஞ்|ட்|ண்|த்|ந்|ப்|ம்|ய்|ர்|ல்|வ்|ழ்|ள்|ற்|ன்]'
        ]
    },
    'en': {
        'words': [
            # Common verbs
            'is', 'are', 'was', 'were', 'will', 'have', 'has', 'had', 'do', 'does', 'did',
            'can', 'could', 'would', 'should', 'may', 'might', 'must', 'shall', 'ought',
            'go', 'come', 'see', 'get', 'make', 'take', 'give', 'know', 'think', 'feel',
            'want', 'need', 'like', 'love', 'hate', 'work', 'play', 'run', 'walk', 'talk',
            'eat', 'drink', 'sleep', 'wake', 'read', 'write', 'listen', 'watch', 'look',
            # Common pronouns
            'I', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
            'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs',
            'this', 'that', 'these', 'those', 'who', 'whom', 'whose', 'which', 'what',
            'myself', 'yourself', 'himself', 'herself', 'itself', 'ourselves', 'themselves',
            # Common prepositions and articles
            'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'of',
            'up', 'down', 'over', 'under', 'above', 'below', 'between', 'among', 'through',
            'during', 'before', 'after', 'since', 'until', 'about', 'around', 'near', 'far',
            'inside', 'outside', 'behind', 'beside', 'against', 'toward', 'towards',
            # Common conjunctions
            'and', 'or', 'but', 'because', 'if', 'then', 'although', 'while', 'since',
            'unless', 'until', 'when', 'where', 'why', 'how', 'whether', 'either', 'neither',
            'both', 'not only', 'as well as', 'however', 'therefore', 'moreover', 'furthermore',
            # Common question words
            'what', 'who', 'where', 'when', 'how', 'why', 'which', 'whose', 'whom',
            # Common adjectives
            'good', 'bad', 'big', 'small', 'new', 'old', 'hot', 'cold', 'long', 'short',
            'tall', 'high', 'low', 'fast', 'slow', 'easy', 'hard', 'light', 'dark', 'heavy',
            'beautiful', 'ugly', 'nice', 'kind', 'mean', 'smart', 'stupid', 'funny', 'serious',
            'happy', 'sad', 'angry', 'excited', 'tired', 'hungry', 'thirsty', 'sick', 'healthy',
            'rich', 'poor', 'young', 'old', 'strong', 'weak', 'clean', 'dirty', 'full', 'empty',
            # Common adverbs
            'very', 'much', 'many', 'few', 'now', 'then', 'also', 'not', 'yes', 'no',
            'here', 'there', 'everywhere', 'somewhere', 'nowhere', 'always', 'never', 'sometimes',
            'often', 'usually', 'rarely', 'today', 'yesterday', 'tomorrow', 'soon', 'late',
            'early', 'quickly', 'slowly', 'carefully', 'loudly', 'quietly', 'well', 'badly',
            # Numbers
            'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
            'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen',
            'eighteen', 'nineteen', 'twenty', 'thirty', 'forty', 'fifty', 'hundred', 'thousand',
            # Time expressions
            'morning', 'afternoon', 'evening', 'night', 'day', 'week', 'month', 'year', 'time', 'hour',
            'minute', 'second', 'moment', 'while', 'period', 'season', 'spring', 'summer', 'fall', 'winter',
            # Common nouns
            'person', 'people', 'man', 'woman', 'child', 'family', 'friend', 'house', 'home', 'school',
            'work', 'job', 'money', 'food', 'water', 'car', 'book', 'phone', 'computer', 'internet'
        ],
        'patterns': [
            # Common word patterns with word boundaries
            r'\b(the|and|that|have|with|this|but|from|they|would|there|been|many|some|time)\b',
            r'\b(which|their|said|each|she|way|make|use|her|could|water|than|first|who)\b',
            r'\b(its|now|find|long|down|day|did|get|come|made|may|part)\b',
            
            # Verb patterns
            r'\b\w+ing\b',  # Present participle
            r'\b\w+ed\b',   # Past tense/past participle
            r'\b\w+s\b',    # Third person singular
            r'\b\w+ly\b',   # Adverbs
            
            # Modal verbs
            r'\b(can|could|will|would|shall|should|may|might|must|ought)\b',
            
            # Auxiliary verbs
            r'\b(is|are|was|were|am|be|being|been)\b',
            r'\b(have|has|had|having)\b',
            r'\b(do|does|did|doing|done)\b',
            
            # Common prefixes
            r'\bun\w+',     # un-
            r'\bre\w+',     # re-
            r'\bpre\w+',    # pre-
            r'\bdis\w+',    # dis-
            r'\bmis\w+',    # mis-
            r'\bover\w+',   # over-
            r'\bunder\w+',  # under-
            r'\bout\w+',    # out-
            r'\bup\w+',     # up-
            
            # Common suffixes
            r'\w+tion\b',   # -tion
            r'\w+sion\b',   # -sion
            r'\w+ness\b',   # -ness
            r'\w+ment\b',   # -ment
            r'\w+able\b',   # -able
            r'\w+ible\b',   # -ible
            r'\w+ful\b',    # -ful
            r'\w+less\b',   # -less
            r'\w+ship\b',   # -ship
            r'\w+hood\b',   # -hood
            
            # Comparative and superlative
            r'\w+er\b',     # -er (comparative)
            r'\w+est\b',    # -est (superlative)
            
            # Question patterns
            r'\b(what|who|where|when|why|how|which|whose)\b.*\?',
            r'\b(is|are|do|does|did|can|could|will|would)\b.*\?',
            
            # Contractions
            r"\b\w+'(t|s|re|ve|ll|d|m)\b",  # Common contractions
            
            # Possessive patterns
            r"\b\w+'s\b",   # Possessive 's
            r"\b\w+s'\b",   # Plural possessive
            
            # Sentence starters
            r'\b(The|A|An|This|That|These|Those|My|Your|His|Her|Our|Their)\b',
            
            # Common English phrases
            r'\b(as well as|in order to|such as|more than|less than|at least|at most)\b',
            r'\b(not only|but also|either or|neither nor|both and)\b',
            
            # Time expressions
            r'\b(in the morning|in the afternoon|in the evening|at night)\b',
            r'\b(last year|next year|this year|every day|every week)\b',
            
            # Frequency adverbs
            r'\b(always|usually|often|sometimes|rarely|never|seldom)\b',
            
            # Intensifiers
            r'\b(very|quite|rather|pretty|fairly|extremely|incredibly|absolutely)\b'
        ]
    }
}

# Supported languages for speech recognition
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

# Language n-gram models
LANGUAGE_NGRAMS = {
    'en': {
        'unigrams': defaultdict(float),
        'bigrams': defaultdict(float),
        'trigrams': defaultdict(float)
    },
    'hi': {
        'unigrams': defaultdict(float),
        'bigrams': defaultdict(float),
        'trigrams': defaultdict(float)
    },
    'ta': {
        'unigrams': defaultdict(float),
        'bigrams': defaultdict(float),
        'trigrams': defaultdict(float)
    }
}

# Pre-computed language statistics
LANGUAGE_STATS = {
    'en': {
        'avg_word_length': 4.7,
        'common_chars': set('etaoinshrdlu'),
        'vowel_ratio': 0.4,
        'consonant_clusters': ['th', 'st', 'ch', 'sh', 'ph', 'wh'],
        'common_endings': ['ing', 'ed', 'ion', 'ity', 'ment', 'ness'],
        'script_ratio': 0.95
    },
    'hi': {
        'avg_word_length': 5.2,
        'common_chars': set('कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह'),
        'vowel_ratio': 0.35,
        'consonant_clusters': ['क्र', 'त्र', 'श्र', 'ज्ञ', 'द्व'],
        'common_endings': ['ता', 'ती', 'ते', 'गा', 'गी', 'गे'],
        'script_ratio': 0.98
    },
    'ta': {
        'avg_word_length': 4.8,
        'common_chars': set('கஙசஞடணதநபமயரலவழளறன'),
        'vowel_ratio': 0.38,
        'consonant_clusters': ['க்ஷ', 'ஸ்ரீ', 'ஜ்ஞ'],
        'common_endings': ['கிற', 'ந்த', 'வ', 'ப்ப', 'ட்'],
        'script_ratio': 0.97
    }
}

def calculate_ngrams(text, n):
    """Calculate n-grams from text"""
    words = text.split()
    ngrams = defaultdict(int)
    for word in words:
        for i in range(len(word) - n + 1):
            ngram = word[i:i+n]
            ngrams[ngram] += 1
    return ngrams

def calculate_language_features(text):
    """Calculate various language features from text"""
    words = text.split()
    chars = ''.join(words)
    
    # Basic statistics
    avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
    
    # Character distribution
    char_freq = defaultdict(int)
    for char in chars:
        char_freq[char] += 1
    
    # Vowel and consonant analysis
    vowels = set('aeiouAEIOU')
    vowel_count = sum(1 for char in chars if char in vowels)
    vowel_ratio = vowel_count / len(chars) if chars else 0
    
    # Script analysis
    script_chars = {
        'en': sum(1 for c in chars if 0x0041 <= ord(c) <= 0x007A),
        'hi': sum(1 for c in chars if 0x0900 <= ord(c) <= 0x097F),
        'ta': sum(1 for c in chars if 0x0B80 <= ord(c) <= 0x0BFF)
    }
    
    # Common patterns
    common_endings = defaultdict(int)
    for word in words:
        if len(word) >= 3:
            common_endings[word[-3:]] += 1
    
    # Consonant clusters
    consonant_clusters = defaultdict(int)
    for word in words:
        for i in range(len(word) - 1):
            if word[i].isalpha() and word[i+1].isalpha():
                consonant_clusters[word[i:i+2]] += 1
    
    return {
        'avg_word_length': avg_word_length,
        'char_freq': dict(char_freq),
        'vowel_ratio': vowel_ratio,
        'script_chars': script_chars,
        'common_endings': dict(common_endings),
        'consonant_clusters': dict(consonant_clusters)
    }

def calculate_language_score(text, lang):
    """Calculate language score using multiple features"""
    features = calculate_language_features(text)
    stats = LANGUAGE_STATS[lang]
    score = 0.0
    weights = {
        'script': 0.4,
        'word_length': 0.1,
        'vowel_ratio': 0.1,
        'endings': 0.2,
        'clusters': 0.1,
        'char_freq': 0.1
    }
    
    # Script score
    script_ratio = features['script_chars'][lang] / len(text) if text else 0
    script_score = 1.0 if abs(script_ratio - stats['script_ratio']) < 0.1 else 0.0
    score += script_score * weights['script']
    
    # Word length score
    word_length_diff = abs(features['avg_word_length'] - stats['avg_word_length'])
    word_length_score = 1.0 if word_length_diff < 0.5 else 0.0
    score += word_length_score * weights['word_length']
    
    # Vowel ratio score
    vowel_ratio_diff = abs(features['vowel_ratio'] - stats['vowel_ratio'])
    vowel_ratio_score = 1.0 if vowel_ratio_diff < 0.1 else 0.0
    score += vowel_ratio_score * weights['vowel_ratio']
    
    # Common endings score
    endings_score = 0.0
    for ending in stats['common_endings']:
        if ending in features['common_endings']:
            endings_score += 1
    endings_score = min(1.0, endings_score / len(stats['common_endings']))
    score += endings_score * weights['endings']
    
    # Consonant clusters score
    clusters_score = 0.0
    for cluster in stats['consonant_clusters']:
        if cluster in features['consonant_clusters']:
            clusters_score += 1
    clusters_score = min(1.0, clusters_score / len(stats['consonant_clusters']))
    score += clusters_score * weights['clusters']
    
    # Character frequency score
    char_freq_score = 0.0
    common_chars = stats['common_chars']
    text_chars = set(features['char_freq'].keys())
    if common_chars and text_chars:
        char_freq_score = len(common_chars.intersection(text_chars)) / len(common_chars)
    score += char_freq_score * weights['char_freq']
    
    return score

def detect_language_from_text(text):
    """New language detection method using statistical analysis"""
    if not text or len(text.strip()) < 2:
        return 'en'
    
    text = text.strip()
    
    # Calculate scores for each language
    scores = {}
    for lang in ['en', 'hi', 'ta']:
        scores[lang] = calculate_language_score(text, lang)
    
    # Get the best matching language
    best_lang = max(scores.items(), key=lambda x: x[1])
    confidence = best_lang[1]
    
    # Store detection details
    st.session_state.last_detection_details = {
        'detected_lang': best_lang[0],
        'confidence': confidence,
        'scores': scores,
        'features': calculate_language_features(text)
    }
    
    # Only return a language if confidence is high enough
    if confidence >= 0.7:  # 70% confidence threshold
        return best_lang[0]
    
    # If confidence is too low, return English as fallback
    st.session_state.last_detection_details['fallback'] = True
    return 'en'

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
                            # Use language detection
                            detected_lang = detect_language_from_text(text)
                            
                            # Get detection details from session state
                            detection_details = st.session_state.get('last_detection_details', {})
                            detection_confidence = detection_details.get('confidence', 0.5)
                            
                            # Calculate total score (recognition success + language match + confidence)
                            lang_match_bonus = 1.0 if detected_lang == lang_code else 0.5
                            total_score = detection_confidence + lang_match_bonus + (len(text.split()) * 0.1)
                            
                            recognition_results.append({
                                'text': text,
                                'recognition_lang': lang_code,
                                'detected_lang': detected_lang,
                                'confidence': detection_confidence,
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
                    
                    # Still run language detection for validation
                    detected_lang = detect_language_from_text(text)
                    
                    # Get detection details from session state
                    detection_details = st.session_state.get('last_detection_details', {})
                    detection_confidence = detection_details.get('confidence', 0.5)
                    
                    st.session_state.last_detection_details = {
                        'recognition_lang': selected_lang,
                        'detected_lang': detected_lang,
                        'confidence': detection_confidence,
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


def send_to_slack(summary, webhook_url, bot_name="Ava"):
    """Ava sends conversation summary to Slack for human agent review"""
    try:
        # Format the message from Ava to the agent
        message = {
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"📬 Message from {bot_name} – Conversation Summary",
                        "emoji": True
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": (
                            f"Hello team! :wave:\n\n"
                            f"I just wrapped up a conversation with a customer. Here's a summary for your review:"
                        )
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"> {summary.replace(chr(10), chr(10) + '> ')}"
                    }
                },
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": f"_Generated on {datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')} by {bot_name}_"
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
        msg['Subject'] = f"{bot_name} – Your Voice Conversation Summary • {datetime.now().strftime('%B %d, %Y at %I:%M %p')}"

        # Add summary to email body
        body = f"""
        <html>
            <body style="font-family: Arial, sans-serif; color: #333;">
                <h2 style="color: #4B0082;">Hi there! I'm {bot_name} 👋</h2>
                <p>I've put together a quick summary of our recent conversation. Here's what we discussed:</p>
                <div style="background-color: #f0f0f5; padding: 15px; border-left: 5px solid #4B0082; border-radius: 6px; margin: 20px 0;">
                    {summary}
                </div>
                <p>If anything feels off or you'd like me to clarify more, I'm always here to help!</p>
                <p style="margin-top: 30px;">Chat recorded on <strong>{datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')}</strong></p>
                <p>With warm regards,</p>
                <p style="font-size: 16px; font-weight: bold;">{bot_name}<br>
                <span style="font-size: 14px; font-weight: normal;">Your Voice Companion</span></p>
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

def get_bearer_token(api_key):
    """Get bearer token for Watsonx API authentication"""
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
    """Get response from Watsonx API"""
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

# Replace the email input field with hardcoded email
email_address = "ananthananth881@gmail.com"  # Replace this with your actual email address

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
    if st.button("Send Summary via Email"):
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
