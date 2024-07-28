import os
import requests
import PyPDF2
import re
import streamlit as st
from dotenv import load_dotenv
from transformers import GPT2Tokenizer

# Load environment variables from .env file
load_dotenv ()

# Initialize tokenizer for token count
tokenizer = GPT2Tokenizer.from_pretrained ("gpt2")


# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader (pdf_file)
    text = ''
    for page_num in range (len (pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text ()
    return text


# Function to preprocess text
def preprocess_text(text):
    text = re.sub (r'\s+', ' ', text)
    text = text.replace ('\n', ' ')
    text = text.strip ()
    return text


# Function to truncate text to fit within the token limit
def truncate_text(text, max_tokens):
    tokens = tokenizer.encode (text)
    if len (tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return tokenizer.decode (tokens)


# Function to get response from Anyscale API
@st.cache_data (show_spinner = False)
def get_response_from_llm(api_key, context, query):
    api_base = "https://api.endpoints.anyscale.com/v1"
    url = f"{api_base}/chat/completions"
    body = {
        "model": "meta-llama/Meta-Llama-3-70B-Instruct",
        "messages": [
            {
                "role": "system",
                "content": f"The user has provided the following text from a PDF document: {context}"
            },
            {
                "role": "user",
                "content": query
            }
        ],
        "temperature": 1,
        "max_tokens": 256,
        "top_p": 1,
        "frequency_penalty": 0
    }

    headers = {"Authorization": f"Bearer {api_key}"}

    with requests.Session () as s:
        response = s.post (url, headers = headers, json = body)
        response_data = response.json ()

    # Extract the assistant's reply from the response
    assistant_reply = response_data.get ("choices", [{}])[0].get ("message", {}).get ("content", "No response")
    return assistant_reply


# Streamlit web interface
def main():
    st.set_page_config (page_title = "Chat with PDF", layout = "wide")

    # Session state to store chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    st.title ("Chatty-patty! 📄🤖")

    st.sidebar.title ("Chat History")
    for chat in st.session_state.chat_history:
        st.sidebar.markdown (f"**Query:** {chat['query']}")
        st.sidebar.markdown (f"**Response:** {chat['response']}")
        st.sidebar.markdown ("---")

    # File uploader for PDF documents
    pdf_file = st.file_uploader ("Upload a PDF", type = "pdf")
    context = ""

    if pdf_file:
        try:
            with st.spinner ("Processing PDF..."):
                # Extract and preprocess text from PDF
                raw_text = extract_text_from_pdf (pdf_file)
                processed_text = preprocess_text (raw_text)

                # Truncate text to fit within the token limit
                max_context_tokens = 8192 - 256  # Reserve tokens for query and other parts
                truncated_text = truncate_text (processed_text, max_context_tokens)
                context = truncated_text
                st.success ("PDF processed successfully. You can now ask questions about its content.")
        except Exception as e:
            st.error (f"An error occurred while processing the PDF: {e}")

    if context:
        # Text input for user queries
        query = st.text_input ("Enter your query about the PDF")

        if query:
            try:
                with st.spinner ("Getting response from LLM..."):
                    # Load API key from environment variable
                    api_key = os.getenv ("ANYSCALE_API_KEY")
                    if not api_key:
                        st.error ("API key not found. Please set the ANYSCALE_API_KEY environment variable.")
                        return

                    # Get response from LLM
                    response = get_response_from_llm (api_key, context, query)

                    # Store chat history in session state
                    st.session_state.chat_history.append ({
                        "query": query,
                        "response": response
                    })

                    # Display chat history and generated response
                    st.write ("**Query:**", query)
                    st.write ("**Response:**", response)
            except Exception as e:
                st.error (f"An error occurred while getting the response: {e}")


if __name__ == "__main__":
    main ()
