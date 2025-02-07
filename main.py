from dotenv import load_dotenv
import requests
import os
import json
import streamlit as st

load_dotenv()

# Contentful Configuration
AUTH_TOKEN = os.getenv("CONTENTFUL_AUTH_TOKEN")
CONTENTFUL_API_URL = "https://graphql.contentful.com/content/v1/spaces/vyqawgh3t5vd"

# Function to query the Contentful API
def query_contentful(query, variables=None):
    headers = {
        "Authorization": AUTH_TOKEN,
        "Content-Type": "application/json"
    }
    response = requests.post(CONTENTFUL_API_URL, headers=headers, json={"query": query, "variables": variables})
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Query failed: {response.status_code}")
        print(response.text)
        return None

# Fetch Master's Programs
def get_masters_programs():
    query = """
    query {
      hecPgeMastersCollection {
        items {
          title
          slug
          description {
            json
          }
        }
      }
    }
    """
    response = query_contentful(query)
    return response["data"]["hecPgeMastersCollection"]["items"]

# Fetch Executive Masters
def get_executive_masters():
    query = """
    query {
      hecPgeExecutiveMastersCollection {
        items {
          title
          slug
          description
          studyFee
          applicationFee
          registrationFee
          modulesCollection {
            items {
              title
            }
          }
        }
      }
    }
    """
    response = query_contentful(query)
    return response["data"]["hecPgeExecutiveMastersCollection"]["items"]

# Fetch Executive Certificates (Filtered)
def get_executive_certificates(title_filter):
    query = """
    query($title: String) {
      hecPgeExecutiveCertificatesCollection(where: { title_contains: $title }) {
        items {
          title
          slug
          description
          studyFee
          applicationFee
          registrationFee
          duration
        }
      }
    }
    """
    variables = {"title": title_filter}
    response = query_contentful(query, variables)
    return response["data"]["hecPgeExecutiveCertificatesCollection"]["items"]

def main():
    print("Fetching Master's Programs...")
    masters_programs = get_masters_programs()
    for program in masters_programs:
        print(f"- {program['title']} (Slug: {program['slug']})")
    
    print("\nFetching Executive Masters...")
    executive_masters = get_executive_masters()
    for program in executive_masters:
        print(f"- {program['title']} (Slug: {program['slug']}, Study Fee: {program['studyFee']})")
    
    print("\nFetching Executive Certificates (Title contains 'log')...")
    certificates = get_executive_certificates("log")
    for cert in certificates:
        print(f"- {cert['title']} (Duration: {cert['duration']}, Study Fee: {cert['studyFee']})")

if __name__ == "__main__":
    st.title("HEC Maroc Chatbot")
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        if message['type'] == 'user':
            st.chat_message('user').markdown(message['text'])
        elif message['type'] == 'bot':
            st.chat_message('bot').markdown(message['text'])
    prompt = st.text_input("How can I help you?")
    masters_programs = get_masters_programs()
    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'type': 'user', 'text': prompt})
        st.session_state.messages.append({'type': 'bot', 'text': masters_programs[0]['title']})
        