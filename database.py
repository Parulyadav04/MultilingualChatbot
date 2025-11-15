import sqlite3
import hashlib
import re
from datetime import datetime
import json
import uuid

class DatabaseManager:
    """Manages user authentication and chat history"""
    
    def __init__(self, db_path="meity_chatbot.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Chat sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
            )
        ''')
        
        # Chat messages table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_messages (
                message_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                language TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id) ON DELETE CASCADE
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def hash_password(self, password):
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def validate_email(self, email):
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def validate_password(self, password):
        """
        Validate password:
        - At least 8 characters
        - At least one uppercase letter
        - At least one lowercase letter
        - At least one special character
        """
        if len(password) < 8:
            return False, "Password must be at least 8 characters long"
        
        if not re.search(r'[A-Z]', password):
            return False, "Password must contain at least one uppercase letter"
        
        if not re.search(r'[a-z]', password):
            return False, "Password must contain at least one lowercase letter"
        
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            return False, "Password must contain at least one special character"
        
        return True, "Valid"
    
    def signup(self, email, password):
        """Create new user account"""
        # Validate email
        if not self.validate_email(email):
            return False, "Invalid email format"
        
        # Validate password
        is_valid, message = self.validate_password(password)
        if not is_valid:
            return False, message
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Check if email already exists
            cursor.execute("SELECT user_id FROM users WHERE email = ?", (email,))
            if cursor.fetchone():
                conn.close()
                return False, "Email already registered"
            
            # Create new user
            user_id = str(uuid.uuid4())
            password_hash = self.hash_password(password)
            
            cursor.execute(
                "INSERT INTO users (user_id, email, password_hash) VALUES (?, ?, ?)",
                (user_id, email, password_hash)
            )
            conn.commit()
            conn.close()
            
            return True, user_id
        
        except Exception as e:
            conn.close()
            return False, f"Error: {str(e)}"
    
    def login(self, email, password):
        """Authenticate user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        password_hash = self.hash_password(password)
        
        cursor.execute(
            "SELECT user_id FROM users WHERE email = ? AND password_hash = ?",
            (email, password_hash)
        )
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return True, result[0]
        return False, "Invalid email or password"
    
    def create_session(self, user_id, title=None):
        """Create new chat session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        session_id = str(uuid.uuid4())
        if not title:
            title = f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        cursor.execute(
            "INSERT INTO chat_sessions (session_id, user_id, title) VALUES (?, ?, ?)",
            (session_id, user_id, title)
        )
        
        conn.commit()
        conn.close()
        
        return session_id
    
    def get_user_sessions(self, user_id):
        """Get all chat sessions for a user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            '''SELECT session_id, title, created_at, updated_at 
               FROM chat_sessions 
               WHERE user_id = ? 
               ORDER BY updated_at DESC''',
            (user_id,)
        )
        
        sessions = cursor.fetchall()
        conn.close()
        
        return [
            {
                'session_id': s[0],
                'title': s[1],
                'created_at': s[2],
                'updated_at': s[3]
            }
            for s in sessions
        ]
    
    def add_message(self, session_id, role, content, language='en'):
        """Add message to chat session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        message_id = str(uuid.uuid4())
        
        cursor.execute(
            '''INSERT INTO chat_messages 
               (message_id, session_id, role, content, language) 
               VALUES (?, ?, ?, ?, ?)''',
            (message_id, session_id, role, content, language)
        )
        
        # Update session timestamp
        cursor.execute(
            "UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
            (session_id,)
        )
        
        conn.commit()
        conn.close()
        
        return message_id
    
    def get_session_messages(self, session_id):
        """Get all messages for a session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            '''SELECT message_id, role, content, language, timestamp 
               FROM chat_messages 
               WHERE session_id = ? 
               ORDER BY timestamp ASC''',
            (session_id,)
        )
        
        messages = cursor.fetchall()
        conn.close()
        
        return [
            {
                'message_id': m[0],
                'role': m[1],
                'content': m[2],
                'language': m[3],
                'timestamp': m[4]
            }
            for m in messages
        ]
    
    def delete_session(self, session_id, user_id):
        """Delete a chat session (with verification)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Verify session belongs to user
        cursor.execute(
            "SELECT user_id FROM chat_sessions WHERE session_id = ?",
            (session_id,)
        )
        result = cursor.fetchone()
        
        if not result or result[0] != user_id:
            conn.close()
            return False
        
        # Delete session (messages will cascade)
        cursor.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
        
        conn.commit()
        conn.close()
        
        return True
    
    def delete_all_sessions(self, user_id):
        """Delete all chat sessions for a user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM chat_sessions WHERE user_id = ?", (user_id,))
        
        conn.commit()
        conn.close()
        
        return True
    
    def update_session_title(self, session_id, title, user_id):
        """Update session title"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Verify session belongs to user
        cursor.execute(
            "SELECT user_id FROM chat_sessions WHERE session_id = ?",
            (session_id,)
        )
        result = cursor.fetchone()
        
        if not result or result[0] != user_id:
            conn.close()
            return False
        
        cursor.execute(
            "UPDATE chat_sessions SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
            (title, session_id)
        )
        
        conn.commit()
        conn.close()
        
        return True
    
    def get_session_context(self, session_id, max_messages=10):
        """Get recent messages as context for LLM"""
        messages = self.get_session_messages(session_id)
        
        # Get last N messages
        recent = messages[-max_messages:] if len(messages) > max_messages else messages
        
        # Format as context
        context = []
        for msg in recent:
            context.append({
                'role': msg['role'],
                'content': msg['content']
            })
        
        return context