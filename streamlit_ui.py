import time
import streamlit as st
from requests_toolbelt.multipart.encoder import MultipartEncoder
import requests
import json
import os

# Constants
FASTAPI_URL = 'http://127.0.0.1:8000'  # Update this as needed
HF_API_KEY_ENV_VAR = 'HUGGINGFACE_API_KEY'

# Function to check if the FastAPI server is running
def is_server_running(url):
    try:
        response = requests.get(url)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

# Check and validate Hugging Face API key
def check_hf_api_key():
    hf_api_key = os.environ.get(HF_API_KEY_ENV_VAR)
    if hf_api_key:
        st.success("Hugging Face API Key is found!")
    else:
        st.error("Failed to update Hugging Face API Key. Please try again.")

    tb_hf_apikey = st.text_input('Hugging Face API Key', hf_api_key)

    return hf_api_key
       
# Wait for the FastAPI server to start
server_message = st.empty()
server_message.info("Waiting for the FastAPI server to start. This should only take a moment...")
while not is_server_running(f'{FASTAPI_URL}/status'):
    time.sleep(1)
server_message.empty()
st.success("FastAPI server is running!")

# Check and validate Hugging Face API key
hf_api_key = check_hf_api_key()

# Streamlit sidebar for user input
st.sidebar.header("Transcription and Diaritization Settings")

task = st.sidebar.selectbox("Task", ["t", "td"])
min_speakers = st.sidebar.slider("Minimum Speakers", 1, 10, 2)
max_speakers = st.sidebar.slider("Maximum Speakers", 1, 10, 2)
if min_speakers > max_speakers:
    st.sidebar.warning("Minimum speakers should not be greater than maximum speakers.")
whisper_model = st.sidebar.selectbox("Whisper Model", ["tiny", "base", "small", "medium", "large", "large-v2"], index=5)
batch_size = st.sidebar.slider("Batch Size", 1, 64, 16)
compute_type = st.sidebar.selectbox("Compute Type", ["float16", "int8"])
dump_model = st.sidebar.checkbox("Dump Model (Clear GPU cache after use)")

# File Uploader
st.header("Upload Audio File")
audio_file = st.file_uploader("Select audio file", type=['wav'])

#TODO: add 

# Process and Display Outputs
if st.button("Process Audio"):
    if hf_api_key is not None:
        if audio_file is not None and min_speakers <= max_speakers:
            # Prepare data for the FastAPI
            request_metadata =  json.dumps({
                        "task": task,
                        "min_speakers": min_speakers,
                        "max_speakers": max_speakers,
                        "whisper_model": whisper_model,
                        "batch_size": batch_size,
                        "compute_type": compute_type,
                        "dump_model": dump_model,
                        "api_key": hf_api_key  # Add the Hugging Face API key to the request
                    })

            multipart_data = MultipartEncoder(
                fields={
                    "file": (audio_file.name, audio_file.getvalue(), "audio/wav"),
                    "settings": request_metadata,
                }
            )

            st.spinner("Processing... This could take a few minutes.")
            
            # Post request to the FastAPI
            response = requests.post(
                f"{FASTAPI_URL}/process_audio",
                data=multipart_data,
                headers={'Content-Type': multipart_data.content_type}
            )

            if response.status_code == 200:
                result = response.json()

                # Display the result
                if result:
                    st.success("Processing complete!")
                    st.balloons()

                    st.header("Results")
                    st.json(result)

                else:
                    st.warning("The result is empty.")
            else:
                st.error(f"Failed to process the audio file. Status code: {response.status_code}. Error: {response.text}")

        else:
            st.error("Please upload an audio file and ensure that the minimum number of speakers is less than or equal to the maximum number of speakers.")
    else:
        st.warning("Hugging Face API Key not found. Please input your API Key.")