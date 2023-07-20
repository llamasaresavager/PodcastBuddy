#!/bin/bash

# Start the FastAPI app
uvicorn app:app --host 0.0.0.0 --port 8000 &

# Start the Streamlit app
streamlit run streamlit_app.py
