from langchain.document_loaders import UnstructuredURLLoader

def load_university_website():
    university_website = "https://www.hec.ac.ma"
    urls = [university_website]
    loader = UnstructuredURLLoader(urls=urls)
    return loader.load()