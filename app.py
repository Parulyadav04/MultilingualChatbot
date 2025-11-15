import streamlit as st
import time
from datetime import datetime
import tempfile
import os
import sounddevice as sd
import soundfile as sf
import numpy as np
import io
import traceback

# Import the chatbot and database
from int_app import IntegratedAPP
from database import DatabaseManager

# Configure page
st.set_page_config(
    page_title="MeitY AI Assistant",
    page_icon="M",  # Removed emoji
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS - New Dark, Simple, and Elegant Theme
st.markdown("""
<style>
    :root {
        --bg-color: #212121;
        --sidebar-bg: #2D2D2D;
        --input-bg: #353535;
        --text-color: #ECECEC;
        --text-color-light: #B0B0B0;
        --border-color: #444444;
        --accent-color: #4a90e2; /* Kept from original for consistency */
        --hover-bg: #3E3E3E;
        --message-bot-bg: #2F2F2F;
        --message-user-bg: #3A3A3A;
        --error-bg: #5a2a2a;
        --error-border: #f44336;
        --info-bg: #2a3a5a;
        --info-border: #2196f3;
        --success-bg: #2a5a2a;
        --success-border: #4caf50;
        --warning-bg: #5a4a2a;
        --warning-border: #ff9800;
    }

    /* General Body */
    body {
        background-color: var(--bg-color);
        color: var(--text-color);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main container */
    .main {
        padding-top: 2rem;
    }
    
    /* --- Sidebar Styling --- */
    [data-testid="stSidebar"] {
        background-color: var(--sidebar-bg);
        border-right: 1px solid var(--border-color);
    }
    [data-testid="stSidebar"] .stButton>button {
        background-color: var(--input-bg);
        border: 1px solid var(--border-color);
        color: var(--text-color);
    }
    [data-testid="stSidebar"] .stButton>button:hover:not(:disabled) {
        background-color: var(--hover-bg);
        border-color: var(--accent-color);
        color: var(--accent-color);
    }
    [data-testid="stSidebar"] .stButton>button[kind="primary"] {
        background: var(--accent-color);
        color: white;
        border: none;
    }
    [data-testid="stSidebar"] .stButton>button[kind="primary"]:hover:not(:disabled) {
        background: #357abd;
    }
    [data-testid="stSidebar"] .stButton>button[kind="primary"]:disabled {
        background: var(--input-bg) !important;
        opacity: 0.5;
    }

    /* Sidebar Title */
    .sidebar-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: var(--text-color);
        margin-bottom: 1rem;
        padding: 0.5rem;
        border-bottom: 2px solid var(--accent-color);
    }
    
    /* Sidebar Chat History Item */
    .chat-history-item {
        padding: 0.5rem 0.2rem;
        border-radius: 6px;
        margin-bottom: 0px;
        transition: all 0.2s;
    }
    .chat-history-item:hover {
        background-color: var(--hover-bg);
    }
    .chat-history-item .stButton>button {
        background: transparent;
        border: none;
        color: var(--text-color);
        text-align: left;
        font-weight: 400;
        font-size: 0.95rem;
        padding-left: 0.5rem;
    }
    .chat-history-item .stButton>button:hover,
    .chat-history-item .stButton>button:focus {
        background: transparent;
        color: var(--text-color);
        border: none;
        box-shadow: none;
    }
    
    /* Popover '...' button */
    [data-testid="stPopover"] {
        background-color: var(--sidebar-bg);
    }
    [data-testid="stPopover"] .stButton>button {
        background: transparent;
        border: none;
        color: var(--text-color-light);
        font-weight: 700;
        font-size: 1.2rem;
    }
    [data-testid="stPopover"] .stButton>button:hover,
    [data-testid="stPopover"] .stButton>button:focus {
        background: var(--hover-bg);
        color: var(--text-color);
        border: none;
        box-shadow: none;
    }
    
    /* Sidebar text/captions */
    [data-testid="stSidebar"] .stCaption {
        color: var(--text-color-light);
        padding-left: 0.7rem;
        margin-top: -8px; /* Pull caption closer to button */
    }
    [data-testid="stSidebar"] .stInfo {
        background-color: transparent;
        color: var(--text-color-light);
        border: none;
        padding-left: 0.5rem;
    }

    /* Custom Scrollbar for dark theme */
    [data-testid="stSidebar"] .simplebar-scrollbar:before {
        background-color: #555;
    }
    [data-testid="stSidebar"] .simplebar-scrollbar.simplebar-visible:before {
        opacity: 0.75;
    }
    [data-testid="stSidebar"] .simplebar-track.simplebar-vertical {
        width: 8px;
    }

    /* --- Auth Page Styling --- */
    .auth-container {
        background: var(--sidebar-bg);
        padding: 2rem;
        border-radius: 8px;
        border: 1px solid var(--border-color);
        max-width: 400px;
        margin: 2rem auto;
    }
    .auth-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text-color);
        text-align: center;
        margin-bottom: 1.5rem;
    }
    /* Tab labels (Login/Sign Up) */
    .stTabs [data-baseweb="tab"] {
        color: var(--text-color-light);
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: var(--accent-color);
    }

    /* --- Main Chat Interface --- */
    
    /* Header */
    .header-container {
        background: var(--sidebar-bg);
        padding: 1.5rem 2rem;
        border-radius: 8px;
        border: 1px solid var(--border-color);
        margin-bottom: 2rem;
        text-align: center;
    }
    .header-title {
        font-size: 1.8rem;
        font-weight: 600;
        color: var(--text-color);
        margin: 0;
    }
    .header-subtitle {
        font-size: 0.95rem;
        color: var(--text-color-light);
        margin-top: 0.5rem;
    }
    
    /* Text Inputs / Text Area */
    .stTextInput, .stTextArea {
        color: var(--text-color);
    }
    .stTextInput label, .stTextArea label {
        color: var(--text-color) !important;
    }
    .stTextInput div[data-baseweb="input"] input,
    .stTextArea textarea {
        background-color: var(--input-bg);
        border: 1px solid var(--border-color);
        color: var(--text-color);
        border-radius: 8px;
        font-size: 1rem;
    }
    .stTextInput div[data-baseweb="input"] input:focus,
    .stTextArea textarea:focus {
        border-color: var(--accent-color);
        box-shadow: 0 0 0 1px var(--accent-color);
        background-color: var(--input-bg);
    }
    
    /* Select box */
    .stSelectbox label {
        color: var(--text-color);
    }
    .stSelectbox div[data-baseweb="select"] {
        background-color: var(--input-bg);
        border: 1px solid var(--border-color);
        color: var(--text-color);
    }
    
    /* Buttons (Main Area) */
    .stButton>button {
        border-radius: 6px;
        font-weight: 500;
        border: 1px solid var(--border-color);
        background: var(--input-bg);
        color: var(--text-color);
        transition: all 0.2s;
    }
    .stButton>button:hover:not(:disabled) {
        border-color: var(--accent-color);
        background: var(--hover-bg);
        color: var(--accent-color);
    }
    .stButton>button[kind="primary"] {
        background: var(--accent-color);
        color: white;
        border: none;
    }
    .stButton>button[kind="primary"]:hover:not(:disabled) {
        background: #357abd;
    }
    .stButton>button:disabled {
        opacity: 0.4;
        cursor: not-allowed;
        background: var(--input-bg) !important;
        color: var(--text-color-light) !important;
        border-color: var(--border-color) !important;
    }
    
    /* Chat Messages */
    .message-container {
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border: 1px solid var(--border-color);
    }
    .user-message {
        background: var(--message-user-bg);
        border-left: 3px solid var(--accent-color);
    }
    .bot-message {
        background: var(--message-bot-bg);
        border-left: 3px solid #34a853;
    }
    .message-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--text-color-light);
        margin-bottom: 0.5rem;
    }
    .message-text {
        font-size: 1rem;
        color: var(--text-color);
        line-height: 1.6;
    }

    /* Status/Info/Error Boxes */
    .status-box {
        padding: 0.8rem;
        border-radius: 6px;
        margin: 1rem 0;
        font-size: 0.9rem;
    }
    .status-info {
        background: var(--info-bg);
        border-left: 3px solid var(--info-border);
        color: #e3f2fd;
    }
    .status-success {
        background: var(--success-bg);
        border-left: 3px solid var(--success-border);
        color: #e8f5e9;
    }
    .status-warning {
        background: var(--warning-bg);
        border-left: 3px solid var(--warning-border);
        color: #fff3e0;
    }
    .status-error {
        background: var(--error-bg);
        border-left: 3px solid var(--error-border);
        color: #ffebee;
    }
    
    /* Recording indicator */
    .recording-box {
        background: var(--warning-bg);
        border: 1px solid var(--warning-border);
        border-radius: 6px;
        padding: 1rem;
        margin: 1rem 0;
        text-align: center;
    }
    .recording-text {
        color: #fff3e0;
        font-weight: 500;
    }

    /* Progress indicator */
    .progress-dots {
        display: inline-block;
        animation: blink 1.4s infinite;
    }
    @keyframes blink {
        0%, 20% { opacity: 0.2; }
        50% { opacity: 1; }
        100% { opacity: 0.2; }
    }
</style>
""", unsafe_allow_html=True)

# Language mapping
LANGUAGE_MAPPING = {
    'en': 'English',
    'hi': 'Hindi (हिन्दी)',
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
    'ur': 'Urdu (اردو)',
    'ne': 'Nepali (नेपाली)',
    'sa': 'Sanskrit (संस्कृत)',
    'mai': 'Maithili (मैथिली)',
    'brx': 'Bodo (बड़ो)',
    'doi': 'Dogri (डोगरी)',
    'ks': 'Kashmiri (कॉशुर)',
    'mni': 'Manipuri (মৈতৈলোন্)',
    'sd': 'Sindhi (سنڌي)'
}

# IndicConformer supported languages
CONFORMER_SUPPORTED_LANGS = {
    'as', 'bn', 'brx', 'doi', 'gu', 'hi', 'kn', 'kok', 
    'ks', 'mai', 'ml', 'mni', 'mr', 'ne', 'or', 'pa', 
    'sa', 'sat', 'sd', 'ta', 'te', 'ur'
}

# Initialize database
@st.cache_resource
def get_database():
    return DatabaseManager()

db = get_database()

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'user_email' not in st.session_state:
    st.session_state.user_email = None
if 'current_session_id' not in st.session_state:
    st.session_state.current_session_id = None
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = None
if 'chatbot_loaded' not in st.session_state:
    st.session_state.chatbot_loaded = False
if 'current_query' not in st.session_state:
    st.session_state.current_query = None
if 'current_query_lang' not in st.session_state:
    st.session_state.current_query_lang = None
if 'current_response' not in st.session_state:
    st.session_state.current_response = None
if 'current_response_lang' not in st.session_state:
    st.session_state.current_response_lang = None
if 'current_audio' not in st.session_state:
    st.session_state.current_audio = None
if 'input_language' not in st.session_state:
    st.session_state.input_language = 'en'
if 'output_language' not in st.session_state:
    st.session_state.output_language = 'en'
if 'error_message' not in st.session_state:
    st.session_state.error_message = None
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'generating_audio' not in st.session_state:
    st.session_state.generating_audio = False
if 'process_query_flag' not in st.session_state:
    st.session_state.process_query_flag = False
if 'pending_query' not in st.session_state:
    st.session_state.pending_query = None
if 'pending_source_lang' not in st.session_state:
    st.session_state.pending_source_lang = None
if 'pending_target_lang' not in st.session_state:
    st.session_state.pending_target_lang = None
if 'text_input_value' not in st.session_state:
    st.session_state.text_input_value = ""
if 'input_key' not in st.session_state:
    st.session_state.input_key = 0
if 'show_signup' not in st.session_state:
    st.session_state.show_signup = False
if 'session_messages' not in st.session_state:
    st.session_state.session_messages = []
if 'show_clear_confirm' not in st.session_state:
    st.session_state.show_clear_confirm = False

@st.cache_resource
def load_chatbot():
    """Load chatbot with lazy loading for STT/TTS"""
    try:
        return IntegratedAPP(
            index_path='meity_faiss.index',
            chunks_path='meity_chunks.json',
            llm_path='Phi-3-mini-4k-instruct-q4.gguf',
            embed_model_name='paraphrase-multilingual-mpnet-base-v2',
            reranker_model_name='cross-encoder/ms-marco-MiniLM-L-6-v2',
            enable_stt=False,
            enable_tts=False,
            enable_translation=True,
            skip_translation=False
        )
    except Exception as e:
        st.error(f"Failed to initialize chatbot: {e}")
        st.stop()

def record_audio(duration=5, sample_rate=16000):
    """Record audio from microphone"""
    audio = sd.rec(
        int(duration * sample_rate), 
        samplerate=sample_rate, 
        channels=1, 
        dtype='float32'
    )
    sd.wait()
    return np.squeeze(audio)

def process_query_sync(query_text, source_lang, target_lang):
    """Process query synchronously"""
    try:
        status_placeholder = st.empty()
        status_placeholder.markdown("""
        <div class="status-box status-info">
            Processing... <span class="progress-dots"></span>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.chatbot.translator:
            st.session_state.chatbot.translator.ensure_models_unloaded()
        
        result = st.session_state.chatbot.ask(
            query_text,
            source_lang=source_lang,
            target_lang=target_lang,
            return_audio=False
        )
        
        st.session_state.current_response = result
        st.session_state.current_response_lang = target_lang
        st.session_state.processing = False
        st.session_state.process_query_flag = False
        
        # Save to database
        if st.session_state.current_session_id:
            db.add_message(
                st.session_state.current_session_id,
                'user',
                query_text,
                source_lang
            )
            db.add_message(
                st.session_state.current_session_id,
                'assistant',
                result['text'],
                target_lang
            )
        
        status_placeholder.empty()
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error details:\n{error_trace}")
        st.session_state.error_message = f"Error: {str(e)}"
        st.session_state.current_response = {
            'text': f"Sorry, I encountered an error. Please try again.",
            'text_en': f"Error: {str(e)}"
        }
        st.session_state.current_response_lang = target_lang
        st.session_state.processing = False
        st.session_state.process_query_flag = False

def load_session_history(session_id):
    """Load messages from a session"""
    messages = db.get_session_messages(session_id)
    st.session_state.session_messages = messages
    
    # Set current query/response to last exchange
    if messages:
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]['role'] == 'assistant':
                st.session_state.current_response = {
                    'text': messages[i]['content'],
                    'text_en': messages[i]['content']
                }
                st.session_state.current_response_lang = messages[i]['language']
                
                # Find corresponding user message
                if i > 0 and messages[i-1]['role'] == 'user':
                    st.session_state.current_query = messages[i-1]['content']
                    st.session_state.current_query_lang = messages[i-1]['language']
                break

# Authentication UI
if not st.session_state.authenticated:
    
    st.title("MeitY AI Assistant")
    st.subheader("Please login or sign up")
    
    # Toggle between login and signup
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    with tab1:
        # st.markdown('<div class="auth-container">', unsafe_allow_html=True)
        # st.markdown('<div class="auth-title">Login</div>', unsafe_allow_html=True)
        
        login_email = st.text_input("Email", key="login_email")
        login_password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login", type="primary", use_container_width=True):
            if login_email and login_password:
                success, result = db.login(login_email, login_password)
                if success:
                    st.session_state.authenticated = True
                    st.session_state.user_id = result
                    st.session_state.user_email = login_email
                    st.success("Login successful!")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error(f"{result}")
            else:
                st.warning("Please enter email and password")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        # st.markdown('<div class="auth-container">', unsafe_allow_html=True)
        # st.markdown('<div class="auth-title">Sign Up</div>', unsafe_allow_html=True)
        
        signup_email = st.text_input("Email", key="signup_email")
        signup_password = st.text_input("Password", type="password", key="signup_password")
        signup_confirm = st.text_input("Confirm Password", type="password", key="signup_confirm")
        
        st.caption("Password must contain:")
        st.caption("• At least 8 characters")
        st.caption("• One uppercase letter")
        st.caption("• One lowercase letter")
        st.caption("• One special character (!@#$%^&*...)")
        
        if st.button("Sign Up", type="primary", use_container_width=True):
            if signup_email and signup_password and signup_confirm:
                if signup_password != signup_confirm:
                    st.error("Passwords do not match")
                else:
                    success, result = db.signup(signup_email, signup_password)
                    if success:
                        st.success("Account created! Please login.")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"{result}")
            else:
                st.warning("Please fill all fields")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.stop()

# Sidebar - Chat History
with st.sidebar:
    st.markdown('<div class="sidebar-title">Chat History</div>', unsafe_allow_html=True)
    
    # New chat button
    if st.button("New Chat", use_container_width=True, type="primary"):
        st.session_state.current_session_id = None
        st.session_state.current_query = None
        st.session_state.current_response = None
        st.session_state.current_audio = None
        st.session_state.session_messages = []
        st.session_state.input_key += 1
        st.rerun()
    
    st.markdown("---")
    
    # Get user's chat sessions
    sessions = db.get_user_sessions(st.session_state.user_id)
    
    if sessions:
        # Scrollable history
        for session in sessions:
            st.markdown(f'<div class="chat-history-item" data-session-id="{session["session_id"]}">', unsafe_allow_html=True)
            col1, col2 = st.columns([0.85, 0.15], gap="small")
            
            with col1:
                # Make session title clickable
                title = session['title'][:30] + "..." if len(session['title']) > 30 else session['title']
                if st.button(
                    title,
                    key=f"session_{session['session_id']}",
                    use_container_width=True,
                    help=session['title']
                ):
                    st.session_state.current_session_id = session['session_id']
                    load_session_history(session['session_id'])
                    st.session_state.current_audio = None
                    st.rerun()
            
            with col2:
                # Popover for delete option
                popover = st.popover("...", use_container_width=True, help="Chat options")
                if popover.button("Delete", key=f"delete_{session['session_id']}", use_container_width=True, type="primary"):
                    db.delete_session(session['session_id'], st.session_state.user_id)
                    if st.session_state.current_session_id == session['session_id']:
                        st.session_state.current_session_id = None
                        st.session_state.current_query = None
                        st.session_state.current_response = None
                        st.session_state.session_messages = []
                    st.rerun()
            
            st.caption(f"{session['updated_at'][:16]}")
            st.markdown('</div>', unsafe_allow_html=True)
            # st.markdown("<hr style='margin: 5px 0; border-color: var(--border-color);'>", unsafe_allow_html=True)

    else:
        st.info("No chat history yet. Start a new conversation!")
    
    st.markdown("---")
    
    # Clear all history with confirmation
    if not st.session_state.show_clear_confirm:
        if st.button("Clear All History", use_container_width=True):
            st.session_state.show_clear_confirm = True
            st.rerun()
    else:
        st.warning("⚠️ This will delete all your chat history!")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✓ Confirm", use_container_width=True, type="primary"):
                db.delete_all_sessions(st.session_state.user_id)
                st.session_state.current_session_id = None
                st.session_state.current_query = None
                st.session_state.current_response = None
                st.session_state.session_messages = []
                st.session_state.show_clear_confirm = False
                st.success("All history cleared!")
                time.sleep(0.5)
                st.rerun()
        with col2:
            if st.button("✗ Cancel", use_container_width=True):
                st.session_state.show_clear_confirm = False
                st.rerun()
    
    st.markdown("---")
    
    # Logout
    if st.button("Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.user_id = None
        st.session_state.user_email = None
        st.session_state.current_session_id = None
        st.session_state.chatbot = None
        st.session_state.chatbot_loaded = False
        st.rerun()

# Main Chat Interface
st.markdown(f"""
<div class="header-container">
    <div class="header-title">MeitY AI Assistant</div>
    <div class="header-subtitle">Welcome, {st.session_state.user_email}</div>
</div>
""", unsafe_allow_html=True)

# Load chatbot
if not st.session_state.chatbot_loaded:
    with st.spinner("Initializing chatbot (Memory-optimized for 8GB RAM)..."):
        try:
            st.session_state.chatbot = load_chatbot()
            st.session_state.chatbot_loaded = True
            time.sleep(0.5)
            st.rerun()
        except Exception as e:
            st.error(f"Failed to load chatbot: {e}")
            st.stop()

# Process pending query
if st.session_state.process_query_flag:
    process_query_sync(
        st.session_state.pending_query,
        st.session_state.pending_source_lang,
        st.session_state.pending_target_lang
    )
    st.rerun()

# Language selection
col1, col2 = st.columns([1, 1])

with col1:
    input_lang = st.selectbox(
        "Input Language",
        options=list(LANGUAGE_MAPPING.keys()),
        format_func=lambda x: LANGUAGE_MAPPING[x],
        index=list(LANGUAGE_MAPPING.keys()).index(st.session_state.input_language),
        key="input_lang_select",
        disabled=st.session_state.processing or st.session_state.generating_audio
    )
    if input_lang != st.session_state.input_language:
        st.session_state.input_language = input_lang

with col2:
    output_lang = st.selectbox(
        "Output Language",
        options=list(LANGUAGE_MAPPING.keys()),
        format_func=lambda x: LANGUAGE_MAPPING[x],
        index=list(LANGUAGE_MAPPING.keys()).index(st.session_state.output_language),
        key="output_lang_select",
        disabled=st.session_state.processing or st.session_state.generating_audio
    )
    if output_lang != st.session_state.output_language:
        st.session_state.output_language = output_lang

st.markdown("---")

# Display audio generation indicator
if st.session_state.generating_audio:
    st.markdown("""
    <div class="status-box status-info">
        Generating audio... <span class="progress-dots">Please wait</span>
    </div>
    """, unsafe_allow_html=True)

# Display error message
if st.session_state.error_message:
    st.markdown(f"""
    <div class="status-box status-error">
        Warning: {st.session_state.error_message}
    </div>
    """, unsafe_allow_html=True)
    st.session_state.error_message = None

# Display current conversation
if st.session_state.current_query and st.session_state.current_query_lang:
    query_lang_label = LANGUAGE_MAPPING[st.session_state.current_query_lang]
    st.markdown(f"""
    <div class="message-container user-message">
        <div class="message-label">You ({query_lang_label})</div>
        <div class="message-text">{st.session_state.current_query}</div>
    </div>
    """, unsafe_allow_html=True)

if st.session_state.current_response and st.session_state.current_response_lang:
    response_data = st.session_state.current_response
    response_lang_label = LANGUAGE_MAPPING[st.session_state.current_response_lang]
    
    if st.session_state.current_response_lang != 'en' and response_data.get('text_en'):
        st.markdown(f"""
        <div class="message-container bot-message">
            <div class="message-label">Assistant (English)</div>
            <div class="message-text">{response_data['text_en']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="message-container bot-message">
        <div class="message-label">Assistant ({response_lang_label})</div>
        <div class="message-text">{response_data['text']}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Speaker button
    col_speaker, col_space = st.columns([1, 5])
    with col_speaker:
        listen_disabled = st.session_state.processing or st.session_state.generating_audio
        
        if st.button("🔊 Listen", key="speaker_button", use_container_width=True, disabled=listen_disabled):
            st.session_state.generating_audio = True
            st.rerun()
    
    # Generate audio
    if st.session_state.generating_audio and not st.session_state.processing:
        if not st.session_state.chatbot.tts_model:
            with st.spinner("Loading text-to-speech model (~1GB)..."):
                try:
                    st.session_state.chatbot.load_tts_model()
                    st.success("TTS loaded!")
                    time.sleep(0.5)
                except Exception as e:
                    st.error(f"Failed to load TTS: {e}")
                    st.session_state.generating_audio = False
                    st.stop()
        
        with st.spinner("Generating audio..."):
            try:
                audio_arr, sample_rate = st.session_state.chatbot.text_to_speech(
                    response_data['text'],
                    st.session_state.current_response_lang
                )
                
                if audio_arr is not None:
                    audio_bytes = io.BytesIO()
                    sf.write(audio_bytes, audio_arr, sample_rate, format='WAV')
                    audio_bytes.seek(0)
                    st.session_state.current_audio = audio_bytes
                    st.success("Audio ready!")
                else:
                    st.error("Failed to generate audio")
                
                st.session_state.generating_audio = False
                time.sleep(0.5)
                st.rerun()
            
            except Exception as e:
                st.error(f"Audio generation error: {e}")
                st.session_state.generating_audio = False
    
    if st.session_state.current_audio:
        st.audio(st.session_state.current_audio, format='audio/wav')

st.markdown("---")

# Input area
if f"user_input_area_{st.session_state.input_key}" not in st.session_state:
    st.session_state[f"user_input_area_{st.session_state.input_key}"] = ""

user_input = st.text_area(
    "Your question",
    height=100,
    placeholder=f"Type your question in {LANGUAGE_MAPPING[st.session_state.input_language]}...",
    key=f"user_input_area_{st.session_state.input_key}",
    label_visibility="collapsed",
    disabled=st.session_state.processing or st.session_state.generating_audio
)

if f"user_input_area_{st.session_state.input_key}" in st.session_state:
    user_input = st.session_state[f"user_input_area_{st.session_state.input_key}"]

# Buttons
col_send, col_voice, col_clear = st.columns([2, 2, 1])

buttons_disabled = st.session_state.processing or st.session_state.generating_audio

with col_send:
    send_button = st.button(
        "Send", 
        use_container_width=True, 
        type="primary",
        disabled=buttons_disabled
    )

with col_voice:
    voice_button = st.button(
        "Voice", 
        use_container_width=True,
        disabled=buttons_disabled
    )

with col_clear:
    clear_button = st.button(
        "Clear", 
        use_container_width=True,
        disabled=buttons_disabled
    )

# Handle voice input
if voice_button:
    if st.session_state.input_language in CONFORMER_SUPPORTED_LANGS:
        loading_msg = f"IndicConformer will load for {LANGUAGE_MAPPING[st.session_state.input_language]}"
    else:
        loading_msg = f"Whisper will load for {LANGUAGE_MAPPING[st.session_state.input_language]}"
    
    st.info(f"{loading_msg}")
    
    st.markdown(f"""
    <div class="recording-box">
        <div class="recording-text">🎙️ Recording 5 seconds in {LANGUAGE_MAPPING[st.session_state.input_language]}...</div>
        <div class="recording-text">Speak clearly after the countdown</div>
    </div>
    """, unsafe_allow_html=True)
    
    countdown_placeholder = st.empty()
    for i in range(3, 0, -1):
        countdown_placeholder.markdown(f"<h3 style='text-align:center;'>{i}</h3>", unsafe_allow_html=True)
        time.sleep(1)
    countdown_placeholder.markdown("<h3 style='text-align:center; color: #e74c3c;'>🔴 Recording...</h3>", unsafe_allow_html=True)
    
    try:
        audio_array = record_audio(duration=5)
        countdown_placeholder.empty()
        
        with st.spinner(f"Converting speech to text using {loading_msg.split()[0]}..."):
            transcription = st.session_state.chatbot.speech_to_text(
                audio_array, 
                st.session_state.input_language
            )
        
        if transcription:
            st.markdown(f"""
            <div class="status-box status-success">
                ✓ Recognized: {transcription}
            </div>
            """, unsafe_allow_html=True)
            
            st.success("STT model unloaded to free memory")
            
            # Create new session if needed
            if not st.session_state.current_session_id:
                session_title = transcription[:50] + "..." if len(transcription) > 50 else transcription
                st.session_state.current_session_id = db.create_session(
                    st.session_state.user_id,
                    session_title
                )
            
            st.session_state.input_key += 1
            st.session_state.current_query = transcription
            st.session_state.current_query_lang = st.session_state.input_language
            st.session_state.current_audio = None
            st.session_state.processing = True
            
            st.session_state.pending_query = transcription
            st.session_state.pending_source_lang = st.session_state.input_language
            st.session_state.pending_target_lang = st.session_state.output_language
            st.session_state.process_query_flag = True
            
            time.sleep(0.5)
            st.rerun()
        else:
            st.error("Could not transcribe audio. Please try again.")
    
    except Exception as e:
        st.error(f"Error during voice input: {e}")

# Handle clear
if clear_button:
    st.session_state.current_query = None
    st.session_state.current_query_lang = None
    st.session_state.current_response = None
    st.session_state.current_response_lang = None
    st.session_state.current_audio = None
    st.session_state.error_message = None
    st.session_state.processing = False
    st.session_state.generating_audio = False
    st.session_state.process_query_flag = False
    st.session_state.pending_query = None
    st.session_state.input_key += 1
    st.rerun()

# Process user input
if send_button and user_input and user_input.strip():
    # Create new session if needed
    if not st.session_state.current_session_id:
        session_title = user_input[:50] + "..." if len(user_input) > 50 else user_input
        st.session_state.current_session_id = db.create_session(
            st.session_state.user_id,
            session_title
        )
    
    st.session_state.input_key += 1
    st.session_state.current_query = user_input
    st.session_state.current_query_lang = st.session_state.input_language
    st.session_state.current_audio = None
    st.session_state.error_message = None
    st.session_state.processing = True
    
    st.session_state.pending_query = user_input
    st.session_state.pending_source_lang = st.session_state.input_language
    st.session_state.pending_target_lang = st.session_state.output_language
    st.session_state.process_query_flag = True
    
    st.rerun()

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #999; font-size: 0.85rem; padding: 1rem;">
    Ask about MeitY<br>
</div>
""", unsafe_allow_html=True)