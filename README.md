# HEC-chatbot

A scalable question-answering chatbot for a university website built using LangChain, OpenAI, and FAISS.

## Features
1. Retrieve static information from the university's website using RAG.
2. Answer user queries in natural language.

## Folder Structure
```
university_chatbot/
├── data/                           # Store raw and processed data
│   └── faiss_index/                # FAISS index files (generated and loaded)
├── loaders/                        # Custom document loaders
│   └── website_loader.py           # Load and preprocess data from the website
├── embeddings/                     # Embedding generation and storage
│   └── create_embeddings.py        # Create embeddings and FAISS index
├── retriever/                      # Retrieval and question-answering logic
│   └── retriever.py                # Query the FAISS index
├── main.py                         # Main entry point to run the chatbot
├── requirements.txt                # Python dependencies
└── README.md                       # Documentation
```

## How to Run
1. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # For Mac/Linux
   venv\Scripts\activate      # For Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create the FAISS index:
   ```bash
   python embeddings/create_embeddings.py
   ```

4. Run the chatbot:
   ```bash
   python main.py
   ```