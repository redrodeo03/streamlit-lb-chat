import streamlit as st
# from openai import OpenAI
import dotenv
import os
from PIL import Image
#from audio_recorder_streamlit import audio_recorder
import base64
from io import BytesIO
# import google.generativeai as genai
import random
import anthropic

dotenv.load_dotenv()


anthropic_models = [
    "claude-3-5-sonnet-20240620"
]

# Function to convert the messages format from OpenAI and Streamlit to Anthropic (the only difference is in the image messages)
def messages_to_anthropic(messages):
    anthropic_messages = []
    prev_role = None
    for message in messages:
        if prev_role and (prev_role == message["role"]):
            anthropic_message = anthropic_messages[-1]
        else:
            anthropic_message = {
                "role": message["role"] ,
                "content": [],
            }
        if message["content"][0]["type"] == "image_url":
            anthropic_message["content"].append(
                {
                    "type": "image",
                    "source":{   
                        "type": "base64",
                        "media_type": message["content"][0]["image_url"]["url"].split(";")[0].split(":")[1],
                        "data": message["content"][0]["image_url"]["url"].split(",")[1]
                        # f"data:{img_type};base64,{img}"
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
def stream_llm_response(model_params, api_key=None):
    response_message = ""
    client = anthropic.Anthropic(api_key=api_key)
    with client.messages.stream(
            model="claude-3-5-sonnet-20240620",
            messages=messages_to_anthropic(st.session_state.messages),
            temperature=model_params["temperature"] if "temperature" in model_params else 0.3,
            max_tokens=4096,
        ) as stream:
            for text in stream.text_stream:
                response_message += text
                yield text

    st.session_state.messages.append({
        "role": "assistant", 
        "content": [
            {
                "type": "text",
                "text": response_message,
            }
        ]})


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

    # --- Side Bar ---
    with st.sidebar:
        cols_keys = st.columns(2)
        default_anthropic_api_key = os.getenv("ANTHROPIC_API_KEY") if os.getenv("ANTHROPIC_API_KEY") is not None else ""
        with st.popover("🔐 Anthropic"):
            anthropic_api_key = st.text_input("Introduce your Anthropic API Key (https://console.anthropic.com/)", value=default_anthropic_api_key, type="password")
    
    # --- Main Content ---
    # Checking if the user has introduced the OpenAI API Key, if not, a warning is displayed
    if (anthropic_api_key == "" or anthropic_api_key is None):
        st.write("#")
        st.warning("⬅️ Please introduce an API Key to continue...")

        with st.sidebar:
            st.write("#")
            st.write("#")
    else:
        client = anthropic.Anthropic(api_key=anthropic_api_key)

        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Displaying the previous messages if there are any
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                for content in message["content"]:
                    if content["type"] == "text":
                        st.write(content["text"])
                    elif content["type"] == "image_url":      
                        st.image(content["image_url"]["url"])
                    elif content["type"] == "video_file":
                        st.video(content["video_file"])
                    elif content["type"] == "audio_file":
                        st.audio(content["audio_file"])

        # Side bar model options and inputs
        with st.sidebar:

            st.divider()
            
            available_models = [] + (anthropic_models if anthropic_api_key else [])
            model = st.selectbox("Select a model:", available_models, index=0)
            model_type = "anthropic"

            model_params = {
                "model": model,
            }

            def reset_conversation():
                if "messages" in st.session_state and len(st.session_state.messages) > 0:
                    st.session_state.pop("messages", None)

            st.button(
                "🗑️ Reset conversation", 
                on_click=reset_conversation,
            )

            st.divider()

            # Image Upload
            if model in ["claude-3-5-sonnet-20240620"]:
                    
                st.write(f"### **🖼️ Add an image{' or a video file' if model_type=='google' else ''}:**")

                def add_image_to_messages():
                    if st.session_state.uploaded_img or ("camera_img" in st.session_state and st.session_state.camera_img):
                        img_type = st.session_state.uploaded_img.type if st.session_state.uploaded_img else "image/jpeg"
                        if img_type == "video/mp4":
                            # save the video file
                            video_id = random.randint(100000, 999999)
                            with open(f"video_{video_id}.mp4", "wb") as f:
                                f.write(st.session_state.uploaded_img.read())
                            st.session_state.messages.append(
                                {
                                    "role": "user", 
                                    "content": [{
                                        "type": "video_file",
                                        "video_file": f"video_{video_id}.mp4",
                                    }]
                                }
                            )
                        else:
                            raw_img = Image.open(st.session_state.uploaded_img or st.session_state.camera_img)
                            img = get_image_base64(raw_img)
                            st.session_state.messages.append(
                                {
                                    "role": "user", 
                                    "content": [{
                                        "type": "image_url",
                                        "image_url": {"url": f"data:{img_type};base64,{img}"}
                                    }]
                                }
                            )

                cols_img = st.columns(2)

                with cols_img[0]:
                    with st.popover("📁 Upload"):
                        st.file_uploader(
                            f"Upload an image{' or a video' if model_type == 'google' else ''}:", 
                            type=["png", "jpg", "jpeg"] + (["mp4"] if model_type == "google" else []), 
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

            # # Audio Upload
            # st.write("#")
            # st.write(f"### **🎤 Add an audio{' (Speech To Text)' if model_type == 'openai' else ''}:**")

            # audio_prompt = None
            # audio_file_added = False
            # if "prev_speech_hash" not in st.session_state:
            #     st.session_state.prev_speech_hash = None

            # speech_input = audio_recorder("Press to talk:", icon_size="3x", neutral_color="#6ca395", )
            # if speech_input and st.session_state.prev_speech_hash != hash(speech_input):
            #     st.session_state.prev_speech_hash = hash(speech_input)
            #     if model_type != "google":
            #         transcript = client.audio.transcriptions.create(
            #             model="whisper-1", 
            #             file=("audio.wav", speech_input),
            #         )

            #         audio_prompt = transcript.text

            #     elif model_type == "google":
            #         # save the audio file
            #         audio_id = random.randint(100000, 999999)
            #         with open(f"audio_{audio_id}.wav", "wb") as f:
            #             f.write(speech_input)

            #         st.session_state.messages.append(
            #             {
            #                 "role": "user", 
            #                 "content": [{
            #                     "type": "audio_file",
            #                     "audio_file": f"audio_{audio_id}.wav",
            #                 }]
            #             }
            #         )

            #         audio_file_added = True

            # st.divider()

        # Chat input
        if prompt := st.chat_input("Hi! Ask me anything..."):
            st.session_state.messages.append(
                    {
                        "role": "user", 
                        "content": [{
                            "type": "text",
                            "text": prompt,
                        }]
                    }
                )
                
                # Display the new messages
            with st.chat_message("user"):
                    st.markdown(prompt)

            with st.chat_message("assistant"):
                st.write_stream(
                    stream_llm_response(
                        model_params=model_params, 
                        api_key=anthropic_api_key
                    )
                )





if __name__=="__main__":
    main()