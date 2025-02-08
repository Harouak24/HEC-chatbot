import os
import json
import pickle
import numpy as np
from services.openai_client import generate_embedding

def cosine_similarity(vec1: list, vec2: list) -> float:
    """
    Computes the cosine similarity between two vectors.
    """
    a = np.array(vec1)
    b = np.array(vec2)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

def rich_text_to_plain_text(rich_text_node: dict) -> str:
    """
    Recursively converts Contentful Rich Text JSON into plain text.
    
    This function handles common node types such as document, paragraph,
    headings, text nodes, unordered/ordered lists, and list items.
    """
    if not rich_text_node:
        return ""
    
    node_type = rich_text_node.get("nodeType", "")
    content = rich_text_node.get("content", [])
    plain_text = ""
    
    if node_type in ["document", "paragraph", "heading-1", "heading-2", 
                     "heading-3", "heading-4", "heading-5", "heading-6", "blockquote"]:
        for child in content:
            plain_text += rich_text_to_plain_text(child) + " "
        plain_text += "\n"
    elif node_type == "text":
        plain_text += rich_text_node.get("value", "")
    elif node_type in ["unordered-list", "ordered-list"]:
        for child in content:
            plain_text += rich_text_to_plain_text(child) + " "
        plain_text += "\n"
    elif node_type == "list-item":
        for child in content:
            plain_text += "- " + rich_text_to_plain_text(child) + "\n"
    else:
        # For any other node types, process their children.
        for child in content:
            plain_text += rich_text_to_plain_text(child) + " "
    
    return plain_text.strip()

def filter_programs_by_query(programs, query):
    """
    Optionally pre-filters programs based on keywords in the query.
    If the query mentions "executive" or "certificate", limit candidates
    to executive masters and executive certificates.
    """
    query_lower = query.lower()
    if "executive" in query_lower or "certificate" in query_lower:
        return [p for p in programs if p.get("type") in ("executive_master", "executive_certificate")]
    return programs

def handle_program_details(query: str,
                           data_file: str = "data/programs.json",
                           embeddings_file: str = "data/program_embeddings.pkl") -> str:
    """
    Uses embeddings matching to find the best program for the user's query,
    then returns a formatted string with that program's details.
    
    For master's programs, the rich-text JSON description is converted to plain text.
    
    Args:
        query (str): The user's query describing the desired program.
        data_file (str): Path to the JSON file with unified program data.
        embeddings_file (str): Path to the pickle file with precomputed embeddings.
    
    Returns:
        str: A formatted string with the program's details, or an error message if no match is found.
    """
    if not query.strip():
        return "Please provide a valid query for program details."
    
    if not os.path.exists(data_file):
        return "No cached program data found. Please run your cache_data script."
    
    if not os.path.exists(embeddings_file):
        return "No precomputed embeddings found. Please run your embeddings script."
    
    # 1. Load program data.
    with open(data_file, "r", encoding="utf-8") as f:
        programs = json.load(f)
    
    # 2. Load precomputed embeddings.
    with open(embeddings_file, "rb") as f:
        embeddings_dict = pickle.load(f)
    
    # 3. Optionally filter candidates based on query keywords.
    candidate_programs = filter_programs_by_query(programs, query)
    if not candidate_programs:
        candidate_programs = programs  # fallback to all programs
    
    # 4. Generate embedding for the query.
    query_embedding = generate_embedding(query)
    if not query_embedding:
        return "Error generating embedding for your query."
    
    # 5. Compute cosine similarities to find the best match.
    best_score = -1.0
    best_slug = None
    for prog in candidate_programs:
        slug = prog.get("slug")
        emb = embeddings_dict.get(slug)
        if emb:
            score = cosine_similarity(query_embedding, emb)
            if score > best_score:
                best_score = score
                best_slug = slug
    
    MIN_SIMILARITY_THRESHOLD = 0.5
    if best_score < MIN_SIMILARITY_THRESHOLD or best_slug is None:
        return "No sufficiently close match was found for your query."
    
    # 6. Retrieve the best matching program.
    matched_program = next((p for p in programs if p.get("slug") == best_slug), None)
    if not matched_program:
        return "The matching program was not found in the cached data."
    
    # 7. Extract details.
    prog_type = matched_program.get("type", "unknown")
    title = matched_program.get("title", "No Title Found")
    description = matched_program.get("description", "")
    study_fee = matched_program.get("studyFee")
    application_fee = matched_program.get("applicationFee")
    registration_fee = matched_program.get("registrationFee")
    duration = matched_program.get("duration")
    modules = matched_program.get("modules", [])
    
    details_lines = [f"Program Details: {title}\n"]
    
    if prog_type == "masters":
        details_lines.append("Type: Master's Program")
        if isinstance(description, dict) and "content" in description:
            plain_text = rich_text_to_plain_text(description)
        else:
            plain_text = str(description)
        details_lines.append(f"Description: {plain_text}")
    
    elif prog_type == "executive_certificate":
        details_lines.append("Type: Executive Certificate")
        if study_fee is not None:
            details_lines.append(f"Study Fee: {study_fee}")
        if application_fee is not None:
            details_lines.append(f"Application Fee: {application_fee}")
        if registration_fee is not None:
            details_lines.append(f"Registration Fee: {registration_fee}")
        if duration:
            details_lines.append(f"Duration: {duration}")
        details_lines.append(f"\nDescription: {description}")
    
    elif prog_type == "executive_master":
        details_lines.append("Type: Executive Master")
        if study_fee is not None:
            details_lines.append(f"Study Fee: {study_fee}")
        if application_fee is not None:
            details_lines.append(f"Application Fee: {application_fee}")
        if registration_fee is not None:
            details_lines.append(f"Registration Fee: {registration_fee}")
        if modules:
            details_lines.append("\nModules:")
            for i, mod in enumerate(modules, start=1):
                details_lines.append(f"  {i}. {mod}")
        details_lines.append(f"\nDescription: {description}")
    
    else:
        details_lines.append("(Unknown Program Type)")
        details_lines.append(f"Description: {description}")
    
    return "\n".join(details_lines)

# Optional test code
if __name__ == "__main__":
    # Example test queries:
    test_query = "I want to know more about the AI master"
    result = handle_program_details(test_query)
    print(result)