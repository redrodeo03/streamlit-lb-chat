import streamlit as st
import dotenv
import os
from PIL import Image
import base64
from io import BytesIO
import random
import anthropic
import json
import hashlib

dotenv.load_dotenv()

# Set the API key directly in the file
ANTHROPIC_API_KEY= st.secrets["ANTHROPIC_API_KEY"] # Replace with your actual API key

anthropic_models = [
    "claude-3-5-sonnet-20240620"
]

# Function to convert the messages format from OpenAI and Streamlit to Anthropic
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

# Function to query and stream the response from the LLM
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

# Function to convert file to base64
def get_image_base64(image_raw):
    buffered = BytesIO()
    image_raw.save(buffered, format=image_raw.format)
    img_byte = buffered.getvalue()
    return base64.b64encode(img_byte).decode('utf-8')

def file_to_base64(file):
    with open(file, "rb") as f:
        return base64.b64encode(f.read())

def base64_to_image(base64_string):
    base64_string = base64_string.split(",")[1]
    return Image.open(BytesIO(base64.b64decode(base64_string)))

def generate_user_id(username):
    return hashlib.md5(username.encode()).hexdigest()

# New function to load user sessions
def load_user_sessions(user_id):
    filename = f"user_{user_id}_sessions.json"
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return {'default': []}

# New function to save user sessions
def save_user_sessions(user_id, sessions):
    filename = f"user_{user_id}_sessions.json"
    with open(filename, "w") as f:
        json.dump(sessions, f)

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

    # User authentication
    username = st.text_input("Enter your username:")
    if username:
        user_id = generate_user_id(username)
        
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
        st.warning("Please enter a username to start chatting.")

if __name__=="__main__":
    main()
