import time
import streamlit as st
from requests_toolbelt.multipart.encoder import MultipartEncoder
import requests
import json
# from transcription.main import format_result
from dotenv import load_dotenv
import os
from os.path import join, dirname
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
import uuid
import pandas as pd
# Constants
FASTAPI_URL = 'http://127.0.0.1:8000'  # Update this as needed

# Function to check if the FastAPI server is running
def is_server_running(url):
    try:
        response = requests.get(url)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

# Wait for the FastAPI server to start
server_message = st.empty()
server_message.info("Waiting for the FastAPI server to start. This should only take a moment...")
while not is_server_running(f'{FASTAPI_URL}/status'):
    time.sleep(1)
server_message.empty()
st.success("FastAPI server is running!")

### CHROMA START ###
db_path = "PodcastBuddy/Database"
default_embeddings = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
# #uses base model and cpu
# # instruct_embeddings_gpu = embedding_functions.InstructorEmbeddingFunction(
# #     model_name="hkunlp/instructor-xl", device="cuda")
client = chromadb.PersistentClient(path=db_path)

def get_create_collection(collection_name):
    collection = client.get_or_create_collection(name=collection_name, embedding_function=default_embeddings)
    return collection

def add_to_collection(collection, text, metadata, ids):
    embeddings = default_embeddings(text)
    # print(embeddings)
    collection.add(
        embeddings=embeddings,
        metadatas=metadata,
        ids=ids,
    )
    return collection

def similar_search_db(collection, query_embeddings, n_results, where):
    collection.query(
        query_embeddings=query_embeddings,
        n_results=n_results,
        where={"style": "style2"}
)

### CHROMA END ###
dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)
# Check and validate Hugging Face API key
hf_api_key = os.getenv("HF_API_KEY_ENV_VAR")
FASTAPI_URL = 'http://127.0.0.1:8000'  # Update this as needed

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

def get_file_size(size_in_bytes):
    # Convert bytes to a human-readable format
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_in_bytes < 1024.0:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024.0

# File Uploader
st.header("Upload Files")
audio_files = st.file_uploader("Select Files", accept_multiple_files=True)
if audio_files:
        file_names = []
        file_sizes = []

        for file in audio_files:
            file_names.append(file.name)
            file_sizes.append(get_file_size(file.size))

        df = pd.DataFrame({"File Name": file_names, "File Size": file_sizes})
        st.dataframe(df)



def combine_speaker_utterances(transcript):
    output_data = []
    current_speaker = None
    current_text = ""
    for obj in transcript:
        speaker, text = obj.popitem()
        
        if speaker == current_speaker:
            current_text += " " + text
        else:
            if current_speaker is not None:
                output_data.append({current_speaker: current_text})
            current_speaker = speaker
            current_text = text

    if current_speaker is not None:
        output_data.append({current_speaker: current_text})
    return output_data

def extract_speaker_and_text(transcription_results):
    for_vector_transcript = []
    for segment in transcription_results["segments"]:
            if "speaker" in segment and "text" in segment:
                new_segment = {
                    segment["speaker"]: segment["text"]
                    # "speaker": segment["speaker"],
                    # "text": segment["text"]
                }
                for_vector_transcript.append(new_segment)
    output_data=combine_speaker_utterances(for_vector_transcript)
        
    return output_data



whole_transcript=""
vector_ts = ""
xmetadata=None

def generate_unique_ids(object_list):
    num_objects = len(object_list)
    unique_ids = [str(uuid.uuid4()) for _ in range(num_objects)]
    print(unique_ids)
    return unique_ids

# Process and Display Outputs
if st.button("Process Audio"):
    if hf_api_key is not None:
        if audio_files is not None and min_speakers <= max_speakers:
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
            
            for audio_file in audio_files:  # Iterate over each file in the list
                # Post request to the FastAPI for each file
                response = requests.post(
                    f"{FASTAPI_URL}/process_audio",
                    data=audio_file.read(),
                    headers={'Content-Type': "audio/wav", "settings": request_metadata}
                )

            if response.status_code == 200:
                result = response.json()
                whole_transcript=result

                # Format and display the result
                if result:
                    st.success("Processing complete!")
                    st.balloons()

                    st.header("Results")
                    fr = extract_speaker_and_text(result)
                    vector_ts = fr
                    # formatted_result = format_result(result, word_segments, include_speaker, include_text, include_time, include_words)
                    st.json(fr)
                    collection = get_create_collection("test")
                    ids = generate_unique_ids(vector_ts)
                    print(ids)
                    add_to_collection(collection, vector_ts, xmetadata, ids)
                    
                    # Download button for the formatted results
                    # st.download_button(label="Download Formatted Results", data=json.dumps(formatted_result).encode(), file_name='formatted_results.json', mime='application/json')

                else:
                    st.warning("The result is empty.")
            else:
                st.error(f"Failed to process the audio file. Status code: {response.status_code}. Error: {response.text}")




# def prep_for_db_upload(vector_ts,):
#     collection = get_create_collection("test")
#     add_to_collection(collection, vector_ts, xmetadata, id)

# if st.button("Upload to Database"):
#     prep_for_db_upload(vector_ts)
    
