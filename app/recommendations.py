import os
import json
import pickle
import numpy as np
from typing import Tuple
from services.openai_client import generate_embedding, generate_completion

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

def load_program_data(data_file: str = "data/programs.json") -> list:
    """
    Loads the unified program data from the JSON cache.
    """
    if not os.path.exists(data_file):
        return []
    with open(data_file, "r", encoding="utf-8") as f:
        programs = json.load(f)
    return programs

def load_embeddings(embeddings_file: str = "data/program_embeddings.pkl") -> dict:
    """
    Loads the precomputed program embeddings.
    """
    if not os.path.exists(embeddings_file):
        return {}
    with open(embeddings_file, "rb") as f:
        embeddings = pickle.load(f)
    return embeddings

def determine_category(background: str) -> Tuple[str, str]:
    """
    Determines the program category to recommend based on the user's background.
    
    Returns:
      - category: "executive_master" or "masters" if determined;
      - error_message: nonempty if the background suggests the user does not hold a bachelor's degree.
    
    Logic:
      - If the background mentions "executive", "senior", or "manager", then category is "executive_master".
      - If the background is empty, default to "masters" (assuming the user holds a bachelor's).
      - If the background mentions "bachelor", then category is "masters".
      - Otherwise, assume the user does not hold a bachelor's and return an error.
    """
    bg_lower = background.lower().strip()
    if bg_lower == "":
        return "masters", ""
    if any(keyword in bg_lower for keyword in ["executive", "senior", "manager"]):
        return "executive_master", ""
    if "bachelor" in bg_lower:
        return "masters", ""
    # If non-empty and no mention of bachelor or executive keywords, assume the user lacks a bachelor.
    return None, ("Our university only offers graduate programs. "
                  "It appears you do not hold a bachelor's degree. "
                  "Please consider obtaining a bachelor's degree first.")

def rank_programs(filtered_programs: list, query_embedding: list, embeddings: dict) -> list:
    """
    Ranks the candidate programs based on cosine similarity with the query embedding.
    
    Returns:
      A list of programs sorted descending by similarity.
    """
    scored = []
    for prog in filtered_programs:
        slug = prog.get("slug")
        prog_emb = embeddings.get(slug)
        if prog_emb:
            score = cosine_similarity(query_embedding, prog_emb)
            scored.append((prog, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [prog for prog, score in scored]

def generate_recommendation_message(background: str, recommended_programs: list) -> str:
    """
    Uses the LLM to generate a creative recommendation message.
    
    The prompt includes the user's background and the list of recommended program titles.
    """
    if not recommended_programs:
        return "No recommended programs found."
    
    list_str = ""
    for idx, prog in enumerate(recommended_programs, start=1):
        title = prog.get("title", "Unknown Program")
        list_str += f"{idx}. {title}\n"
    
    prompt = (
        f"You are a creative academic advisor. The user provided the following background: \"{background}\".\n"
        f"Based on this, I recommend the following programs:\n{list_str}\n"
        "Please provide a creative recommendation message explaining why these programs might be an excellent fit for the user. "
        "Include some insights on how these programs align with the user's background and future aspirations."
    )
    
    # Use a high temperature for creativity.
    recommendation = generate_completion(prompt, model="gpt-3.5-turbo", max_tokens=250, temperature=0.9)
    return recommendation

def recommend_programs(background: str,
                         data_file: str = "data/programs.json",
                         embeddings_file: str = "data/program_embeddings.pkl") -> str:
    """
    Recommends programs based on the user's background.
    
    Steps:
      1. Determine the recommendation category (masters vs. executive_master) based on background.
         - If the user is an executive (keywords present), use "executive_master".
         - If background is empty or mentions "bachelor", use "masters".
         - Otherwise, if it appears the user does not hold a bachelor's, return an advisory message.
      2. Load program data and precomputed embeddings.
      3. Filter programs by the determined category.
      4. If background is provided, generate an embedding for the background and rank the candidate programs.
         Otherwise, list all programs in that category.
      5. Use a creative LLM prompt (with high temperature) to produce a recommendation message.
    
    Returns:
      A creative recommendation message string.
    """
    category, error_message = determine_category(background)
    if error_message:
        return error_message
    
    programs = load_program_data(data_file)
    if not programs:
        return "No program data available."
    
    # Filter the programs by the determined category.
    filtered_programs = [p for p in programs if p.get("type") == category]
    if not filtered_programs:
        return f"No programs available in the {category} category."
    
    embeddings = load_embeddings(embeddings_file)
    
    # If the background is non-empty, rank programs using embedding similarity.
    if background.strip():
        bg_embedding = generate_embedding(background)
        if not bg_embedding:
            return "Error generating embedding for your background."
        ranked_programs = rank_programs(filtered_programs, bg_embedding, embeddings)
        # Select top 3 recommendations.
        recommended_programs = ranked_programs[:3] if ranked_programs else filtered_programs
    else:
        # If no background provided, simply recommend (up to) 3 programs from the category.
        recommended_programs = filtered_programs[:3]
    
    # Generate and return a creative recommendation message.
    return generate_recommendation_message(background, recommended_programs)

# Optional test code
if __name__ == "__main__":
    # Case 1: User holds a bachelor's and is looking for a master's (background mentions bachelor).
    test_bg_1 = "I hold a Bachelor's in Business Administration and want to advance my career in finance."
    print("Test 1:")
    print(recommend_programs(test_bg_1))
    print("\n-----------------\n")
    
    # Case 2: Background empty; assume user holds a bachelor's.
    test_bg_2 = ""
    print("Test 2:")
    print(recommend_programs(test_bg_2))
    print("\n-----------------\n")
    
    # Case 3: User is an executive.
    test_bg_3 = "I am an executive with 15 years of experience leading multinational teams."
    print("Test 3:")
    print(recommend_programs(test_bg_3))
    print("\n-----------------\n")
    
    # Case 4: User does not hold a bachelor's.
    test_bg_4 = "I have completed high school and want to further my education."
    print("Test 4:")
    print(recommend_programs(test_bg_4))
    print("\n-----------------\n")