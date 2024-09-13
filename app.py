import sys
import streamlit as st
import os
from PIL import Image
import base64
from io import BytesIO
import anthropic
import hashlib
import dotenv
from google.oauth2 import id_token
from google.auth.transport import requests
from google_auth_oauthlib.flow import Flow
import logging
import sqlite3
import json

logging.basicConfig(level=logging.INFO)
dotenv.load_dotenv()

# SQLite initialization
def get_db_connection():
    conn = sqlite3.connect('chat_sessions.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS sessions
        (user_id TEXT, session_name TEXT, messages TEXT,
         PRIMARY KEY (user_id, session_name))
    ''')
    conn.commit()
    conn.close()

init_db()

# Set the API key directly in the file (consider using environment variables in production)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Google OAuth Configuration
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
REDIRECT_URI = "http://localhost:8501"  # Update this for production

anthropic_models = [
    "claude-3-5-sonnet-20240620"
]

# Configure Google OAuth flow
flow = Flow.from_client_config(
    {
        "web": {
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    },
    scopes=["openid", "https://www.googleapis.com/auth/userinfo.email", "https://www.googleapis.com/auth/userinfo.profile"],
    redirect_uri=REDIRECT_URI,
)

def verify_google_token(token):
    try:
        request = requests.Request()
        clock_skew_in_seconds = 300
        idinfo = id_token.verify_oauth2_token(
            token, 
            request, 
            GOOGLE_CLIENT_ID,
            clock_skew_in_seconds=clock_skew_in_seconds
        )
        if idinfo['iss'] not in ['accounts.google.com', 'https://accounts.google.com']:
            raise ValueError('Wrong issuer.')
        logging.info(f"Token verification successful. ID info: {idinfo}")
        if 'email' not in idinfo:
            logging.error("Email not found in ID token.")
            return None
        return idinfo
    except ValueError as e:
        logging.error(f"Token verification failed: {str(e)}")
        return None

def is_valid_email(email):
    logging.info(f"Validating email: {email}")
    is_valid = email.lower().endswith("@upsurge.io")
    logging.info(f"Is email valid: {is_valid}")
    return is_valid

def login():
    if 'google_token' not in st.session_state:
        auth_url, _ = flow.authorization_url(prompt="consent")
        st.markdown(
            f"""
    <a href="{auth_url}" target="_self" style="
        display: inline-block;
        background-color: #6CA395;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        text-decoration: none;
        font-weight: bold;
        text-align: center;
        width: 100%;
        box-sizing: border-box;
        font-family: Arial, sans-serif;
    ">
        Sign in with Google
    </a>
            """,
            unsafe_allow_html=True
        )
        return False
    else:
        idinfo = verify_google_token(st.session_state.google_token)
        if idinfo:
            if 'email' in idinfo:
                email = idinfo['email']
                if is_valid_email(email):
                    st.success(f"Logged in as {email}")
                    return True
                else:
                    st.error(f"Unauthorized email: {email}. Please log in with a valid @upsurge.io account.")
            else:
                st.error("Email not found in authentication response. Please ensure you've granted email access permission.")
        else:
            st.error("Failed to verify Google authentication token. Please try logging in again.")
        
        if 'google_token' in st.session_state:
            del st.session_state.google_token
        return False

def messages_to_anthropic(messages):
    anthropic_messages = []
    prev_role = None
    for message in messages:
        if prev_role and (prev_role == message["role"]):
            anthropic_message = anthropic_messages[-1]
        else:
            anthropic_message = {
                "role": message["role"],
                "content": [],
            }
        if message["content"][0]["type"] == "image_url":
            anthropic_message["content"].append(
                {
                    "type": "image",
                    "source": {   
                        "type": "base64",
                        "media_type": message["content"][0]["image_url"]["url"].split(";")[0].split(":")[1],
                        "data": message["content"][0]["image_url"]["url"].split(",")[1]
                    }
                }
            )
        else:
            anthropic_message["content"].append(message["content"][0])

        if prev_role != message["role"]:
            anthropic_messages.append(anthropic_message)

        prev_role = message["role"]
        
    return anthropic_messages

def stream_llm_response(model_params, messages):
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    with client.messages.stream(
            model="claude-3-5-sonnet-20240620",
            messages=messages_to_anthropic(messages),
            temperature=model_params.get("temperature", 0.3),
            max_tokens=4096,
        ) as stream:
            for text in stream.text_stream:
                yield text

def get_image_base64(image_raw):
    buffered = BytesIO()
    image_raw.save(buffered, format=image_raw.format)
    img_byte = buffered.getvalue()
    return base64.b64encode(img_byte).decode('utf-8')

def generate_user_id(email):
    return hashlib.md5(email.encode()).hexdigest()

# Load user sessions from SQLite
def load_user_sessions(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT session_name, messages FROM sessions WHERE user_id = ?", (user_id,))
    sessions = {row['session_name']: json.loads(row['messages']) for row in cursor.fetchall()}
    conn.close()
    return sessions if sessions else {'default': []}

# Save user sessions to SQLite
def save_user_sessions(user_id, sessions):
    conn = get_db_connection()
    cursor = conn.cursor()
    for session_name, messages in sessions.items():
        cursor.execute('''
            INSERT OR REPLACE INTO sessions (user_id, session_name, messages)
            VALUES (?, ?, ?)
        ''', (user_id, session_name, json.dumps(messages)))
    conn.commit()
    conn.close()

def main():
    # --- Page Config ---
    st.set_page_config(
        page_title="lb-chat",
        page_icon="☁️",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    # --- Header ---
    st.markdown("""<h1 style="text-align: center; color: #6ca395;">☁️ <i>{{upsurge}} chat</h1>""", unsafe_allow_html=True)

    # Check for OAuth callback
    if 'code' in st.query_params:
        code = st.query_params['code']
        flow.fetch_token(code=code)
        st.session_state.google_token = flow.credentials.id_token
        st.query_params.clear()

    if login():
        # Get user info
        idinfo = verify_google_token(st.session_state.google_token)
        user_email = idinfo['email']
        user_id = generate_user_id(user_email)
        
        # Initialize session state for managing multiple chat sessions
        if 'sessions' not in st.session_state:
            st.session_state.sessions = load_user_sessions(user_id)
        if 'current_session' not in st.session_state:
            st.session_state.current_session = 'default'

        # --- Side Bar ---
        with st.sidebar:
            # Session management
            st.subheader("Session Management")
            session_options = list(st.session_state.sessions.keys())
            selected_session = st.selectbox("Select a session:", session_options, index=session_options.index(st.session_state.current_session))
            
            if selected_session != st.session_state.current_session:
                st.session_state.current_session = selected_session
                st.rerun()

            if st.button("Create New Session"):
                new_session_id = f"Session_{len(st.session_state.sessions) + 1}"
                st.session_state.sessions[new_session_id] = []
                st.session_state.current_session = new_session_id
                save_user_sessions(user_id, st.session_state.sessions)
                st.rerun()

        # --- Main Content ---
        # Use the current session's messages
        messages = st.session_state.sessions[st.session_state.current_session]

        # Displaying the previous messages if there are any
        for message in messages:
            with st.chat_message(message["role"]):
                for content in message["content"]:
                    if content["type"] == "text":
                        st.markdown(content["text"])
                    elif content["type"] == "image_url":      
                        st.image(content["image_url"]["url"])

        # Side bar model options and inputs
        with st.sidebar:
            st.divider()
            
            model = anthropic_models[0]  # Use the first model by default
            model_type = "anthropic"

            model_params = {
                "model": model,
            }

            def reset_conversation():
                st.session_state.sessions[st.session_state.current_session] = []
                save_user_sessions(user_id, st.session_state.sessions)
                st.rerun()

            st.button(
                "🗑️ Reset conversation", 
                on_click=reset_conversation,
            )

            st.divider()

            # Image Upload
            st.write("### **🖼️ Add an image:**")

            def add_uploaded_image_to_messages():
                if st.session_state.uploaded_img:
                    img_type = st.session_state.uploaded_img.type
                    raw_img = Image.open(st.session_state.uploaded_img)
                    img = get_image_base64(raw_img)
                    st.session_state.sessions[st.session_state.current_session].append(
                        {
                            "role": "user", 
                            "content": [{
                                "type": "image_url",
                                "image_url": {"url": f"data:{img_type};base64,{img}"}
                            }]
                        }
                    )
                    save_user_sessions(user_id, st.session_state.sessions)
                    st.rerun()

            def add_camera_image_to_messages():
                if 'camera_img' in st.session_state and st.session_state.camera_img is not None:
                    img_type = "image/jpeg"
                    raw_img = Image.open(st.session_state.camera_img)
                    img = get_image_base64(raw_img)
                    st.session_state.sessions[st.session_state.current_session].append(
                        {
                            "role": "user", 
                            "content": [{
                                "type": "image_url",
                                "image_url": {"url": f"data:{img_type};base64,{img}"}
                            }]
                        }
                    )
                    save_user_sessions(user_id, st.session_state.sessions)
                    st.session_state.reset_camera = True
                    st.rerun()


            # File uploader
            st.write("#### Upload an image:")
            uploaded_file = st.file_uploader(
                "Choose an image file", 
                type=["png", "jpg", "jpeg"],
                key="uploaded_img",
            )
            if uploaded_file:
                if st.button("Add uploaded image to chat"):
                    add_uploaded_image_to_messages()

            # Camera input
            st.write("#### Or use camera:")
            if 'camera_active' not in st.session_state:
                st.session_state.camera_active = False
            if 'reset_camera' not in st.session_state:
                st.session_state.reset_camera = False

            if not st.session_state.camera_active:
                if st.button("Activate camera"):
                    st.session_state.camera_active = True
                    st.session_state.reset_camera = False
                    st.rerun()
            elif st.session_state.reset_camera:
                st.session_state.camera_active = False
                st.session_state.reset_camera = False
                st.rerun()
            else:
                camera_image = st.camera_input("Take a picture", key="camera_img")
                if camera_image is not None:
                    if st.button("Add camera image to chat"):
                        add_camera_image_to_messages()
                
                if st.button("Deactivate camera"):
                    st.session_state.camera_active = False
                    st.session_state.reset_camera = True
                    st.rerun()

        # Chat input
        if prompt := st.chat_input("Hi! Ask me anything..."):
            st.session_state.sessions[st.session_state.current_session].append(
                {
                    "role": "user", 
                    "content": [{
                        "type": "text",
                        "text": prompt,
                    }]
                }
            )
            save_user_sessions(user_id, st.session_state.sessions)
                
            # Display the new user message
            with st.chat_message("user"):
                st.markdown(prompt)

            # Display the assistant's response
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_response = ""
                for chunk in stream_llm_response(
                    model_params=model_params,
                    messages=st.session_state.sessions[st.session_state.current_session]
                ):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)

            # Add the assistant's response to the current session
            st.session_state.sessions[st.session_state.current_session].append(
                {
                    "role": "assistant",
                    "content": [{
                        "type": "text",
                        "text": full_response
                    }]
                }
            )
            save_user_sessions(user_id, st.session_state.sessions)

if __name__=="__main__":
    main()
