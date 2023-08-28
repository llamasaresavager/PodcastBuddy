import time
import streamlit as st
from requests_toolbelt.multipart.encoder import MultipartEncoder
import requests
import json
# from transcription.main import format_result

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



