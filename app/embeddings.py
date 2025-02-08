import os
import json
import pickle
from services.openai_client import generate_embedding

def precompute_program_embeddings(
    data_file: str = "data/programs.json", 
    output_file: str = "data/program_embeddings.pkl"
):
    """
    Precomputes embeddings for each program in programs.json and saves them to a pickle file.

    Args:
        data_file (str): Path to the JSON file containing consolidated program data.
        output_file (str): Path to the pickle file where embeddings will be stored.

    The output file contains a dictionary mapping each program's 'slug' to its embedding vector.
    """
    # 1. Load consolidated program data
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"{data_file} does not exist. Please run your caching script first.")

    with open(data_file, "r", encoding="utf-8") as f:
        programs = json.load(f)

    # 2. Generate embeddings
    embeddings_dict = {}
    for program in programs:
        slug = program.get("slug", "unknown_slug")
        title = program.get("title", "")
        # description can be string or dict depending on the program type
        desc = program.get("description", "")

        # Convert dict-based descriptions (e.g., master's JSON) to string if needed
        if isinstance(desc, dict):
            # Typically, if you want to flatten or convert to text, you'd parse the dict.
            # For now, let's just convert the whole dict to a JSON string.
            desc = json.dumps(desc)

        # Combine title + description as the text for embeddings
        combined_text = f"{title}\n{desc}"

        embedding = generate_embedding(combined_text)
        embeddings_dict[slug] = embedding

    # 3. Save embeddings to a pickle file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "wb") as f:
        pickle.dump(embeddings_dict, f)

    print(f"Embeddings saved to {output_file}")

if __name__ == "__main__":
    # Run this script directly to generate embeddings
    precompute_program_embeddings()