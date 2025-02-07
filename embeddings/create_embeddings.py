from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from loaders.website_loader import load_university_website

def create_faiss_index():
    documents = load_university_website()
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local("data/faiss_index")
    print("FAISS index created and saved!")

if __name__ == "__main__":
    create_faiss_index()