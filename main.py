import requests
import PyPDF2
import re
from dotenv import load_dotenv
from transformers import GPT2Tokenizer
import streamlit as st

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
