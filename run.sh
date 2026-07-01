#!/bin/bash

# 1. Boot the FastAPI backend server in the background on port 8000
# The '&' character tells Linux to run this process as an independent background worker
uvicorn backend.main:app --host 0.0.0.0 --port 8000 &

# 2. Wait for 5 seconds to give the heavy FastEmbed models a moment to warm up in memory
sleep 5

# 3. Boot the Streamlit frontend client dashboard in the foreground on port 7860
# Note: Port 7860 is the strict default port required by Hugging Face Spaces to render your UI
streamlit run frontend/app.py --server.port 7860 --server.address 0.0.0.0
