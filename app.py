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
from google.auth import jwt
import time
from google.auth.transport import requests
from google_auth_oauthlib.flow import Flow
import logging
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import json

logging.basicConfig(level=logging.INFO)
dotenv.load_dotenv()

# Firebase initialization
firebase_creds = dict(st.secrets["FIREBASE_CREDENTIALS"])

# Ensure the private_key is properly formatted
firebase_creds['private_key'] = firebase_creds['private_key'].replace('\\n', '\n')

if not firebase_admin._apps:
    cred = credentials.Certificate(firebase_creds)
    firebase_admin.initialize_app(cred)
db = firestore.client()

# Set the API key directly in the file (consider using environment variables in production)
ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"]
GOOGLE_CLIENT_ID = st.secrets["GOOGLE_CLIENT_ID"]
GOOGLE_CLIENT_SECRET = st.secrets["GOOGLE_CLIENT_SECRET"]


# Determine the base URL dynamically
REDIRECT_URI = "https://lb-chat.streamlit.app/callback"

anthropic_models = [
    "claude-3-5-sonnet-20240620"
]

# Configure Google OAuth flow
flow = Flow.from_client_config(
    {
        "web": {
            "client_id": os.environ.get("GOOGLE_CLIENT_ID"),
            "client_secret": os.environ.get("GOOGLE_CLIENT_SECRET"),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    },
    scopes=["openid",
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/userinfo.profile"],
    redirect_uri=REDIRECT_URI,
)

def verify_google_token(token):
    try:
        request = requests.Request()
        clock_skew_in_seconds = 300
        idinfo = id_token.verify_oauth2_token(
            token, 
            request, 
            os.environ.get('GOOGLE_CLIENT_ID'),
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
        
        # Create a Google login button
        st.markdown(
            f"""
            <br>
            <a href="{auth_url}" target="_self">
                <div style="
                    display: centre-align;
                    background-color: #6CA394;
                    color: white;
                    padding: 10px 20px;
                    border-radius: 5px;
                    text-decoration: none;
                    font-weight: bold;
                    text-align: center;
                    transition: background-color 0.3s;
                ">
                    <img src="https://cdn1.iconfinder.com/data/icons/google-s-logo/150/Google_Icons-09-512.png" 
                         style="vertical-align: middle; height: 24px; margin-right: 10px;" style="text-decoration:none;">
                    Sign in with Google
                </div>
            </a>
            """,
            unsafe_allow_html=True
        )
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
            temperature=model_params["temperature"] if "temperature" in model_params else 0.3,
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

# Load user sessions from Firebase
def load_user_sessions(user_id):
    sessions = {}
    docs = db.collection('users').document(user_id).collection('sessions').stream()
    for doc in docs:
        sessions[doc.id] = doc.to_dict().get('messages', [])
    return sessions if sessions else {'default': []}

# Save user sessions to Firebase
def save_user_sessions(user_id, sessions):
    user_ref = db.collection('users').document(user_id)
    for session_name, messages in sessions.items():
        user_ref.collection('sessions').document(session_name).set({'messages': messages})

def main():
    # --- Page Config ---
    st.set_page_config(
        page_title="lb-chat",
        page_icon="☁️",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    # --- Header ---
    st.markdown("""<h1 style="text-align: center; color: #6ca395;">☁️ <i>{{lemmebuild}} chat</h1>""", unsafe_allow_html=True)

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

            def add_image_to_messages():
                if st.session_state.uploaded_img or ("camera_img" in st.session_state and st.session_state.camera_img):
                    img_type = st.session_state.uploaded_img.type if st.session_state.uploaded_img else "image/jpeg"
                    raw_img = Image.open(st.session_state.uploaded_img or st.session_state.camera_img)
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

            cols_img = st.columns(2)

            with cols_img[0]:
                with st.popover("📁 Upload"):
                    st.file_uploader(
                        "Upload an image:", 
                        type=["png", "jpg", "jpeg"],
                        accept_multiple_files=False,
                        key="uploaded_img",
                        on_change=add_image_to_messages,
                    )

            with cols_img[1]:                    
                with st.popover("📸 Camera"):
                    activate_camera = st.checkbox("Activate camera")
                    if activate_camera:
                        st.camera_input(
                            "Take a picture", 
                            key="camera_img",
                            on_change=add_image_to_messages,
                        )

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
    else:
        st.warning("Please log in to use the chat application.")

if __name__=="__main__":
    main()
