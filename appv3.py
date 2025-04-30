import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load the environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Initialize the Gemini LLM via LangChain
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=gemini_api_key)

# Streamlit App UI
st.title("🔍 Ask Me Anything (Gemini + LangChain)")
st.write("Test your Gemini API key and LangChain integration")

question = st.text_input("Enter your question:")

if st.button("Get Answer") and question:
    try:
        with st.spinner("Getting answer..."):
            response = llm.invoke(question)
            st.success("Response received!")
            st.write(response.content)
    except Exception as e:
        st.error(f"❌ Error: {e}")
