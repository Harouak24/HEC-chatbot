import streamlit as st
from ..app.orchestrator import chatbot_orchestrator

def main():
    # Configure the page
    st.set_page_config(page_title="HEC Chatbot", layout="wide")
    
    # Title and instructions
    st.title("HEC Chatbot")
    st.markdown("Ask me anything about our programs, and I'll help you find the right fit!")
    
    # Input box for the user's query
    user_query = st.text_input("Enter your query here:")
    
    # When the user clicks "Submit"
    if st.button("Submit"):
        if not user_query.strip():
            st.error("Please enter a valid query.")
        else:
            with st.spinner("Generating response..."):
                response = chatbot_orchestrator(user_query)
            st.markdown("### Chatbot Response:")
            st.write(response)

if __name__ == "__main__":
    main()