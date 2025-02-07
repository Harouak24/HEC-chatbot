from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def query_faiss(query):
    API_KEY = "OPENAI_API_KEY"
    embeddings = OpenAIEmbeddings(openai_api_key = API_KEY)
    vectorstore = FAISS.load_local("data/faiss_index", embeddings)
    qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0), retriever=vectorstore.as_retriever())
    return qa_chain.run(query)

if __name__ == "__main__":
    query = "What programs are offered by HEC?"
    response = query_faiss(query)
    print(f"Response: {response}")