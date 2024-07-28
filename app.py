from main import extract_text_from_pdf, preprocess_text, truncate_text, get_response_from_llm
import streamlit as st
import os


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
