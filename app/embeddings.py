import os
import json
import pickle
from services.openai_client import generate_embedding

def rich_text_to_plain_text(rich_text_node: dict) -> str:
    """
    Recursively converts Contentful Rich Text JSON into a plain-text string.
    Handles paragraphs, text nodes, lists, etc.

    Args:
        rich_text_node (dict): A node from the "json" part of the Contentful Rich Text.

    Returns:
        str: The plain-text extracted from the Rich Text node.
    """
    if not rich_text_node:
        return ""

    node_type = rich_text_node.get("nodeType", "")
    content = rich_text_node.get("content", [])

    # Accumulate child text
    plain_text = ""

    if node_type in ["document", "paragraph", "heading-1", "heading-2", "heading-3",
                     "heading-4", "heading-5", "heading-6", "blockquote"]:
        # These node types contain further content
        for child in content:
            plain_text += rich_text_to_plain_text(child)
        # Separate paragraphs/blocks with a space or newline if you like
        plain_text += "\n"

    elif node_type == "text":
        # Actual text node
        plain_text += rich_text_node.get("value", "")

    elif node_type in ["unordered-list", "ordered-list"]:
        # For lists, process each list item
        for child in content:
            plain_text += rich_text_to_plain_text(child)
        plain_text += "\n"

    elif node_type == "list-item":
        # Each list item might contain paragraphs or text
        for child in content:
            plain_text += "- " + rich_text_to_plain_text(child) + "\n"

    else:
        # If there are other node types, handle or skip them
        # We'll just recurse into 'content' if available
        for child in content:
            plain_text += rich_text_to_plain_text(child)

    return plain_text.strip()

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
    if not os.path.exists(data_file):
        raise FileNotFoundError(
            f"{data_file} does not exist. Please run your caching script (cache_data.py) first."
        )

    # 1. Load consolidated program data
    with open(data_file, "r", encoding="utf-8") as f:
        programs = json.load(f)

    # 2. Generate embeddings
    embeddings_dict = {}
    for program in programs:
        slug = program.get("slug", "unknown_slug")
        title = program.get("title", "")
        desc_value = program.get("description", "")

        # If this is a master's program, desc_value might be a dict: {"json": {...}}
        # For others (exec masters/certs), it's often just a string
        if isinstance(desc_value, dict):
            # Parse the "json" key to extract plain text
            rich_text = desc_value.get("json", {})
            plain_text = rich_text_to_plain_text(rich_text)
        else:
            # Already a plain string (Executive Master/Certificate)
            plain_text = desc_value or ""

        # Combine title + parsed description for embedding
        combined_text = f"{title}\n{plain_text}"
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