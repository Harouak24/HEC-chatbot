import os
import json
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter

def cache_hec_webpages():
    """
    Loads content from three HEC web pages, splits the text by the period ('.')
    to approximate sentence boundaries, and caches the resulting document chunks
    as JSON.
    """
    urls = [
        "https://hec.ac.ma",
        "https://www.hec.ac.ma/executive-education",
        "https://www.hec.ac.ma/executive-certificates",
        "https://www.hec.ac.ma/executive-masters/executive-master-gestion-de-projets",
        "https://www.hec.ac.ma/executive-masters/executive-master-finance",
        "https://www.hec.ac.ma/executive-masters/executive-master-data-gouvernance-cloud-computing-cybersecurite",
        "https://www.hec.ac.ma/executive-masters/executive-master-data-analytics-machine-learning",
        "https://www.hec.ac.ma/executive-masters/executive-master-intelligence-artificielle-big-data",
        "https://www.hec.ac.ma/executive-masters/executive-master-qualite-hygiene-securite-environnement-qhse",
        "https://www.hec.ac.ma/executive-masters/executive-master-management-et-ressources-humaines",
        "https://www.hec.ac.ma/executive-masters/executive-master-achats-logistique-et-supply-chain-management",
        "https://www.hec.ac.ma/executive-masters/executive-master-audit-et-controle-de-gestion",
        "https://www.hec.ac.ma/executive-masters/executive-master-marketing-digital-e-business",
        "https://www.hec.ac.ma/programme-grande-ecole",
        "https://www.hec.ac.ma/programme-grande-ecole/intelligence-artificielle-management",
        "https://www.hec.ac.ma/programme-grande-ecole/logistique-et-commerce-international",
        "https://www.hec.ac.ma/programme-grande-ecole/comptabilite-controle-et-audit",
        "https://www.hec.ac.ma/programme-grande-ecole/marketing-e-business-digital-strategy",
        "https://www.hec.ac.ma/programme-grande-ecole/finance"
    ]
    
    all_docs = []
    
    # Define a text splitter that splits by period.
    splitter = CharacterTextSplitter(separator=".", chunk_size=1500, chunk_overlap=300)
    
    for url in urls:
        print(f"Loading content from: {url}")
        loader = WebBaseLoader(url)
        docs = loader.load()  # This returns a list of Document objects.
        # Split each Document's content into smaller chunks.
        split_docs = splitter.split_documents(docs)
        # Optionally, add the URL to the document's metadata for traceability.
        for doc in split_docs:
            doc.metadata["source"] = url
        all_docs.extend(split_docs)
    
    # Ensure the data directory exists.
    os.makedirs("data", exist_ok=True)
    output_file = os.path.join("data", "hec_webpages.json")
    
    # Convert Document objects to dicts for JSON serialization.
    docs_as_dicts = [
        {"page_content": doc.page_content, "metadata": doc.metadata}
        for doc in all_docs
    ]
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(docs_as_dicts, f, ensure_ascii=False, indent=2)
    
    print(f"Cached {len(all_docs)} document chunks to {output_file}")

if __name__ == "__main__":
    cache_hec_webpages()