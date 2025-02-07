from retriever.retriever import query_faiss

def main():
    print("Welcome to the University Chatbot!")
    while True:
        query = input("Ask a question: ")
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        response = query_faiss(query)
        print(f"Answer: {response}")

if __name__ == "__main__":
    main()