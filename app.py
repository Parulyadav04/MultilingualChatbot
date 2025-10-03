import streamlit as st
import speech_recognition as sr
import pyttsx3
from gtts import gTTS
import io
import base64
from langdetect import detect
import tempfile
import os
from datetime import datetime
import pandas as pd
from streamlit_chat import message
import time

# Configure page
st.set_page_config(
    page_title="Multilingual AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .chat-container {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    
    .language-badge {
        display: inline-block;
        background: #667eea;
        color: white;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin-bottom: 0.5rem;
    }
    
    .stats-container {
        background: #f8f9ff;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .voice-button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        padding: 0.8rem 1.5rem;
        border-radius: 25px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .voice-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    .sidebar-content {
        background: #f8f9ff;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Language mapping for Indian languages
LANGUAGE_MAPPING = {
    'hi': 'Hindi (हिन्दी)',
    'en': 'English',
    'bn': 'Bengali (বাংলা)',
    'te': 'Telugu (తెలుగు)',
    'ta': 'Tamil (தமிழ்)',
    'mr': 'Marathi (मराठी)',
    'gu': 'Gujarati (ગુજરાતી)',
    'kn': 'Kannada (ಕನ್ನಡ)',
    'ml': 'Malayalam (മലയാളം)',
    'pa': 'Punjabi (ਪੰਜਾਬੀ)',
    'or': 'Odia (ଓଡ଼ିଆ)',
    'as': 'Assamese (অসমীয়া)',
    'ur': 'Urdu (اردو)'
}

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'language_stats' not in st.session_state:
    st.session_state.language_stats = {}

def detect_language(text):
    """Detect language of input text"""
    try:
        detected_lang = detect(text)
        return detected_lang if detected_lang in LANGUAGE_MAPPING else 'en'
    except:
        return 'en'

def text_to_speech(text, language='en'):
    """Convert text to speech and return audio bytes"""
    try:
        tts = gTTS(text=text, lang=language, slow=False)
        fp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        tts.save(fp.name)
        
        with open(fp.name, 'rb') as audio_file:
            audio_bytes = audio_file.read()
        
        os.unlink(fp.name)
        return audio_bytes
    except Exception as e:
        st.error(f"Error generating audio: {str(e)}")
        return None

def speech_to_text():
    """Convert speech to text"""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Listening... Speak now!")
        try:
            # Listen for audio with timeout
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            st.info("🔄 Processing your speech...")
            
            # Try to recognize speech in multiple languages
            text = recognizer.recognize_google(audio)
            return text
        except sr.WaitTimeoutError:
            st.warning("⏱️ Listening timeout. Please try again.")
            return None
        except sr.UnknownValueError:
            st.warning("🤷‍♂️ Could not understand the audio. Please speak clearly.")
            return None
        except sr.RequestError as e:
            st.error(f"❌ Speech recognition error: {str(e)}")
            return None

def simulate_chatbot_response(user_input, detected_lang):
    """Simulate chatbot response - Replace with your actual chatbot logic"""
    # This is where you would integrate your actual chatbot/AI model
    responses = {
        'hi': f"आपका सवाल '{user_input}' के लिए धन्यवाद। मैं एक बहुभाषी चैटबॉट हूं और आपकी सहायता के लिए यहां हूं।",
        'en': f"Thank you for your question '{user_input}'. I'm a multilingual chatbot here to help you.",
        'bn': f"আপনার প্রশ্ন '{user_input}' এর জন্য ধন্যবাদ। আমি একটি বহুভাষিক চ্যাটবট এবং আপনাকে সাহায্য করতে এখানে আছি।",
        'te': f"మీ ప్రశ్న '{user_input}' కు ధన్যవాదాలు. నేను బహుభాషా చాట్‌బాట్ మరియు మీకు సహాయం చేయడానికి ఇక్కడ ఉన్నాను।",
        'ta': f"உங்கள் கேள்வி '{user_input}' க்கு நன்றி. நான் ஒரு பலமொழி சாட்பாட் மற்றும் உங்களுக்கு உதவ இங்கே இருக்கிறேன்।"
    }
    
    return responses.get(detected_lang, responses['en'])

def update_language_stats(language):
    """Update language usage statistics"""
    if language in st.session_state.language_stats:
        st.session_state.language_stats[language] += 1
    else:
        st.session_state.language_stats[language] = 1

# Main UI
st.markdown("""
<div class="main-header">
    <h1> Multilingual AI Chatbot</h1>
    <p>Speak or type in your preferred Indian language - I understand them all!</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.header("🌐 Language Support")
    st.write("**Supported Languages:**")
    for code, name in LANGUAGE_MAPPING.items():
        st.write(f"• {name}")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # # Language Statistics
    # if st.session_state.language_stats:
    #     st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    #     st.header("Usage Statistics")
    #     for lang_code, count in st.session_state.language_stats.items():
    #         lang_name = LANGUAGE_MAPPING.get(lang_code, lang_code)
    #         st.write(f"**{lang_name}:** {count} messages")
    #     st.markdown("</div>", unsafe_allow_html=True)
    
    # Settings
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.header("⚙️ Settings")
    enable_auto_speech = st.checkbox("Auto-play responses", value=True)
    voice_speed = st.slider("Voice Speed", 0.5, 2.0, 1.0, 0.1)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Clear Chat
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# Main chat interface
col1, col2 = st.columns([3, 1])

with col1:
    # Input methods
    st.subheader("💬 Chat Input")
    
    # Text input
    user_input = st.text_area(
        "Type your message:",
        height=100,
        placeholder="Type your message in any supported language..."
    )
    
    # Button row
    col_send, col_voice, col_clear = st.columns([1, 1, 1])
    
    with col_send:
        send_button = st.button("📤 Send Message", use_container_width=True)
    
    with col_voice:
        voice_button = st.button("🎤 Voice Input", use_container_width=True)
    
    with col_clear:
        clear_input = st.button("🧹 Clear Input", use_container_width=True)

with col2:
    st.subheader("🎵 Voice Controls")
    st.info("Click 'Voice Input' to speak your message, or type and click 'Send Message'")

# Handle voice input
if voice_button:
    with st.spinner("🎤 Listening for your voice..."):
        voice_text = speech_to_text()
        if voice_text:
            user_input = voice_text
            st.success(f"Voice recognized: {voice_text}")

# Handle clear input
if clear_input:
    user_input = ""
    st.rerun()

# Process user input
if send_button and user_input.strip():
    # Detect language
    detected_lang = detect_language(user_input)
    detected_lang_name = LANGUAGE_MAPPING.get(detected_lang, f"Unknown ({detected_lang})")
    
    # Update statistics
    update_language_stats(detected_lang)
    
    # Add user message to chat history
    st.session_state.chat_history.append({
        'type': 'user',
        'message': user_input,
        'language': detected_lang,
        'timestamp': datetime.now().strftime("%H:%M:%S")
    })
    
    # Generate bot response
    with st.spinner("🤔 Thinking..."):
        bot_response = simulate_chatbot_response(user_input, detected_lang)
    
    # Add bot response to chat history
    st.session_state.chat_history.append({
        'type': 'bot',
        'message': bot_response,
        'language': detected_lang,
        'timestamp': datetime.now().strftime("%H:%M:%S")
    })
    
    # Generate audio for bot response
    if enable_auto_speech:
        with st.spinner("🔊 Generating audio response..."):
            audio_bytes = text_to_speech(bot_response, detected_lang)
            if audio_bytes:
                st.audio(audio_bytes, format='audio/mp3', autoplay=True)
    
    st.rerun()

# Display chat history
if st.session_state.chat_history:
    st.subheader("💭 Chat History")
    
    for i, chat in enumerate(reversed(st.session_state.chat_history[-10:])):  # Show last 10 messages
        lang_name = LANGUAGE_MAPPING.get(chat['language'], chat['language'])
        
        if chat['type'] == 'user':
            st.markdown(f"""
            <div class="chat-container" style="margin-left: 20%; background: #e3f2fd;">
                <div class="language-badge">👤 You - {lang_name}</div>
                <p><strong>{chat['message']}</strong></p>
                <small style="color: #666;">{chat['timestamp']}</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            col_msg, col_audio = st.columns([4, 1])
            
            with col_msg:
                st.markdown(f"""
                <div class="chat-container" style="margin-right: 20%; background: #f3e5f5;">
                    <div class="language-badge" style="background: #764ba2;">🤖 Bot - {lang_name}</div>
                    <p>{chat['message']}</p>
                    <small style="color: #666;">{chat['timestamp']}</small>
                </div>
                """, unsafe_allow_html=True)
            
            with col_audio:
                if st.button(f"🔊", key=f"audio_{i}", help="Play audio"):
                    with st.spinner("Generating audio..."):
                        audio_bytes = text_to_speech(chat['message'], chat['language'])
                        if audio_bytes:
                            st.audio(audio_bytes, format='audio/mp3', autoplay=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🌟 Multilingual AI Chatbot | Built with Streamlit | Supports Indian Languages</p>
</div>
""", unsafe_allow_html=True)

# Real-time language detection preview
if user_input.strip():
    detected = detect_language(user_input)
    lang_name = LANGUAGE_MAPPING.get(detected, detected)
    st.sidebar.success(f"🌍 Detected Language: {lang_name}")



# import streamlit as st
# from chatbot import MeityChatbot
# import time

# # Initialize chatbot
# @st.cache_resource
# def load_chatbot():
#     return MeityChatbot()

# def main():
#     st.set_page_config(
#         page_title="MeitY AI Assistant",
#         page_icon="🏛️",
#         layout="wide"
#     )
    
#     st.title("🏛️ MeitY AI Assistant")
#     st.markdown("Ask questions about Ministry of Electronics and Information Technology policies, schemes, and notifications.")
    
#     # Load chatbot
#     try:
#         chatbot = load_chatbot()
#     except Exception as e:
#         st.error(f"Failed to load chatbot: {e}")
#         st.info("Please make sure you've run `python process_data.py` first to process your data.")
#         return
    
#     # Chat interface
#     if "messages" not in st.session_state:
#         st.session_state.messages = []
    
#     # Display chat history
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])
    
#     # Chat input
#     if prompt := st.chat_input("Ask about MeitY policies, schemes, or notifications..."):
#         # Add user message
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         with st.chat_message("user"):
#             st.markdown(prompt)
        
#         # Generate response
#         with st.chat_message("assistant"):
#             with st.spinner("Searching documents and generating response..."):
#                 result = chatbot.chat(prompt)
                
#                 # Only show the answer, not sources
#                 st.markdown(result['response'])
                
#                 # Add assistant message
#                 st.session_state.messages.append({
#                     "role": "assistant", 
#                     "content": result['response']
#                 })
    
#     # Sidebar with information
#     with st.sidebar:
#         st.header("About")
#         st.markdown("""
#         This AI assistant helps you find information from MeitY documents including:
#         - Government policies
#         - Scheme notifications  
#         - Ministry updates
#         - Technical guidelines
#         """)
        
#         if st.button("Clear Chat History"):
#             st.session_state.messages = []
#             st.rerun()
        
#         st.header("Example Questions")
#         example_questions = [
#             "What is C-DAC?",
#             "Contact number of MeitY",
#             "What is Digital India initiative?",
#             "Tell me about electronics manufacturing schemes",
#             "What are the latest IT policies?"
#         ]
        
#         for question in example_questions:
#             if st.button(question, key=f"example_{question[:20]}"):
#                 st.session_state.messages.append({"role": "user", "content": question})
#                 st.rerun()

# if __name__ == "__main__":
#     main()
