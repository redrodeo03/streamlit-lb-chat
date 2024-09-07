import streamlit as st
import os
from PIL import Image
import base64
from io import BytesIO
import anthropic
import json
import hashlib

# Set the API key directly in the file or preferably in Streamlit secrets
ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"]

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

def generate_user_id(username):
    return hashlib.md5(username.encode()).hexdigest()

# New function to load user sessions using Streamlit secrets
def load_user_sessions(user_id):
    sessions_json = st.secrets.get(f"user_{user_id}", "{}")
    return json.loads(sessions_json)

# New function to save user sessions using Streamlit secrets
def save_user_sessions(user_id, sessions):
    sessions_json = json.dumps(sessions)
    st.secrets[f"user_{user_id}"] = sessions_json

def main():
    st.set_page_config(
        page_title="lb-chat",
        page_icon="☁️",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    st.markdown("""<h1 style="text-align: center; color: #6ca395;">☁️ <i>{{lemmebuild}} chat</h1>""", unsafe_allow_html=True)

    username = st.text_input("Enter your username:")
    if username:
        user_id = generate_user_id(username)
        
        if 'sessions' not in st.session_state:
            st.session_state.sessions = load_user_sessions(user_id)
        if 'current_session' not in st.session_state:
            st.session_state.current_session = 'default'

        with st.sidebar:
            st.subheader("Session Management")
            
            for session_name in st.session_state.sessions.keys():
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"• {session_name}")
                with col2:
                    if st.button("Select", key=f"select_{session_name}", use_container_width=True):
                        st.session_state.current_session = session_name
                        st.rerun()
                with col3:
                    if st.button("Rename", key=f"rename_{session_name}", use_container_width=True):
                        new_name = st.text_input(f"New name for {session_name}:", key=f"new_name_{session_name}")
                        if new_name and new_name != session_name:
                            st.session_state.sessions[new_name] = st.session_state.sessions.pop(session_name)
                            if st.session_state.current_session == session_name:
                                st.session_state.current_session = new_name
                            save_user_sessions(user_id, st.session_state.sessions)
                            st.rerun()

            if st.button("Create New Session", use_container_width=True):
                new_session_id = f"Session_{len(st.session_state.sessions) + 1}"
                st.session_state.sessions[new_session_id] = []
                st.session_state.current_session = new_session_id
                save_user_sessions(user_id, st.session_state.sessions)
                st.rerun()

        messages = st.session_state.sessions[st.session_state.current_session]

        for message in messages:
            with st.chat_message(message["role"]):
                for content in message["content"]:
                    if content["type"] == "text":
                        st.markdown(content["text"])
                    elif content["type"] == "image_url":      
                        st.image(content["image_url"]["url"])

        with st.sidebar:
            st.divider()
            
            model = anthropic_models[0]
            model_params = {"model": model}

            def reset_conversation():
                st.session_state.sessions[st.session_state.current_session] = []
                save_user_sessions(user_id, st.session_state.sessions)
                st.rerun()

            st.button("🗑️ Reset conversation", on_click=reset_conversation, use_container_width=True)

            st.divider()

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
                
            with st.chat_message("user"):
                st.markdown(prompt)

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
