import streamlit as st
from requests_toolbelt.multipart.encoder import MultipartEncoder
import requests
import json

# Streamlit sidebar for user input
st.sidebar.header("Transcription and Diaritization Settings")

task = st.sidebar.selectbox("Task", ["t", "td"])
min_speakers = st.sidebar.slider("Minimum Speakers", 1, 10, 2)
max_speakers = st.sidebar.slider("Maximum Speakers", 1, 10, 2)
whisper_model = st.sidebar.selectbox("Whisper Model", ["tiny", "base", "small", "medium", "large", "large-v2"], index=5)
batch_size = st.sidebar.slider("Batch Size", 1, 64, 16)
compute_type = st.sidebar.selectbox("Compute Type", ["float16", "int8"])
dump_model = st.sidebar.checkbox("Dump Model (Clear GPU cache after use)")

# File Uploader
st.header("Upload Audio File")
audio_file = st.file_uploader("Select audio file", type=['wav'])

# Process and Display Outputs
if st.button("Process Audio"):
    if audio_file is not None:
        # Prepare data for the FastAPI
        multipart_data = MultipartEncoder(
            fields={
                "file": (audio_file.name, audio_file.getvalue(), "audio/wav"),
                "settings": ('settings', json.dumps({
                    "task": task,
                    "min_speakers": min_speakers,
                    "max_speakers": max_speakers,
                    "whisper_model": whisper_model,
                    "batch_size": batch_size,
                    "compute_type": compute_type,
                    "dump_model": dump_model
                }), 'application/json')
            }
        )

        # Post request to the FastAPI
        response = requests.post(
            "http://localhost:8000/process_audio",  # Replace with your actual FastAPI URL
            data=multipart_data,
            headers={'Content-Type': multipart_data.content_type}
        )

        if response.status_code == 200:
            result = response.json()
            st.spinner("Processing... This could take a few minutes.")
            
            # Display the result
            if result:
                st.success("Processing complete!")
                st.balloons()

                st.header("Results")
                st.json(result)

            else:
                st.warning("The result is empty.")
        else:
            st.error(f"Failed to process the audio file. Error: {response.text}")

    else:
        st.error("Please upload an audio file.")
