import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
gemini_api_key = st.secrets["GEMINI_API_KEY"]

st.title(gemini_api_key[0:7])
