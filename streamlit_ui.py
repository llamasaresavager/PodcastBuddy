import time
import streamlit as st
from requests_toolbelt.multipart.encoder import MultipartEncoder
import requests
import json
import os

# Constants
FASTAPI_URL = 'http://127.0.0.1:8000'  # Update this as needed

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

def format_result(result, include_word_segments, include_speaker, include_text, include_time, include_words):
    formatted_result = {}

    if include_word_segments and 'word_segments' in result:
        formatted_result["word_segments"] = result["word_segments"]

    if 'segments' in result:
        formatted_result["segments"] = []
        for segment in result["segments"]:
            seg_dict = {}

            if include_text and 'text' in segment:
                seg_dict['text'] = segment['text']

            if include_time and 'start' in segment and 'end' in segment:
                seg_dict['time'] = f"{segment['start']}-{segment['end']}"

            if include_words and 'words' in segment:
                seg_dict['words'] = segment['words']

            if include_speaker and 'speaker' in segment:
                seg_dict['speaker'] = segment['speaker']

            formatted_result["segments"].append(seg_dict)

    return formatted_result


# Wait for the FastAPI server to start
server_message = st.empty()
server_message.info("Waiting for the FastAPI server to start. This should only take a moment...")
while not is_server_running(f'{FASTAPI_URL}/status'):
    time.sleep(1)
server_message.empty()
st.success("FastAPI server is running!")

# Check and validate Hugging Face API key
hf_api_key = "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"

# Streamlit sidebar for user input
st.sidebar.header("Transcription and Diaritization Settings")
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

# Format Results
st.sidebar.header("Format Results")
word_segments = st.sidebar.checkbox("Word Segments")
st.sidebar.markdown("---")  # Add a horizontal rule
include_speaker = st.sidebar.checkbox("Speaker")
include_text = st.sidebar.checkbox("Text")
include_time = st.sidebar.checkbox("Time Range")
include_words = st.sidebar.checkbox("Words")

# Process and Display Outputs
if st.button("Process Audio"):
    if hf_api_key is not None:
        if audio_file is not None and min_speakers <= max_speakers:
            # Prepare data for the FastAPI
            request_metadata =  json.dumps({
                        "min_speakers": min_speakers,
                        "max_speakers": max_speakers,
                        "whisper_model": whisper_model,
                        "batch_size": batch_size,
                        "compute_type": compute_type,
                        "dump_model": dump_model,
                        "api_key": hf_api_key  # Add the Hugging Face API key to the request
                    })
            st.spinner("Processing... This could take a few minutes.")
            
            # Post request to the FastAPI
            response = requests.post(
                f"{FASTAPI_URL}/process_audio",
                data=audio_file.read(),
                headers={'Content-Type': "audio/wav", "settings": request_metadata}
            )

            if response.status_code == 200:
                result = response.json()

                # Format and display the result
                if result:
                    st.success("Processing complete!")
                    st.balloons()

                    st.header("Results")
                    formatted_result = format_result(result, word_segments, include_speaker, include_text, include_time, include_words)
                    st.json(formatted_result)
                    
                    # Download button for the formatted results
                    st.download_button(label="Download Formatted Results", data=json.dumps(formatted_result).encode(), file_name='formatted_results.json', mime='application/json')

                else:
                    st.warning("The result is empty.")
            else:
                st.error(f"Failed to process the audio file. Status code: {response.status_code}. Error: {response.text}")