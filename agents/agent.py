import os
import json
import pickle
import numpy as np
from openai import openai
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


def cosine_similarity(vec1: list, vec2: list) -> float:
    a = np.array(vec1)
    b = np.array(vec2)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

def generate_embedding(text: str) -> list:
    """
    Generates an embedding vector for the given text using OpenAI's API.

    Args:
        text (str): The text to embed.

    Returns:
        list: A 1D list of floats representing the embedding vector.
              Returns an empty list if there's an error.
    """
    try:
        response = openai.embeddings.create(
            model="text-embedding-3-small",  # or "text-embedding-ada-002", etc.
            input=text,
            encoding_format="float"
        )
        # 'response' is a typed object
        embedding = response.data[0].embedding
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return []

def get_all_masters_tool(query: str) -> str:
    """
    Returns a grouped list of all university masters programs.
    """
    with open("data/programs.json", "r", encoding="utf-8") as f:
        programs = json.load(f)
    masters = []
    for prog in programs:
        prog_type = prog.get("type")
        title = prog.get("title")
        if prog_type == "masters":
            masters.append(title)
    output += f"HEC Masters:\n"
    for i, m in enumerate(masters, start=1):
        output += f"  {i}. {m}\n"
    output += "\n"
    return output.strip()

def get_all_executive_masters_tool(query: str) -> str:
    """
    Returns a grouped list of all university executive masters programs.
    """
    with open("data/programs.json", "r", encoding="utf-8") as f:
        programs = json.load(f)
    grouped = []
    for prog in programs:
        prog_type = prog.get("type")
        title = prog.get("title")
        modules = prog.get("modules")
        if prog_type == "executive_master":
            grouped.append([title, modules])
    output += f"HEC Executive Masters:\n"
    i = 1
    while i <= len(grouped):
        title, modules = grouped[i-1]
        output += f"  {i}. {title}\n"
        if modules:
            output += "    Certificates:\n"
            for j, mod in enumerate(modules, start=1):
                output += f"      {j}. {mod}\n"
        i += 1
    return output.strip()

def get_program_details_tool(query: str) -> str:
    """
    Uses embedding matching to find the best program for a given query,
    then returns its detailed information. This method is robust to typos
    or incomplete program names.
    """
    data_file = "data/programs.json"
    embeddings_file = "data/program_embeddings.pkl"
    
    if not os.path.exists(data_file):
        return "Program data not found."
    if not os.path.exists(embeddings_file):
        return "Program embeddings not found."
    
    with open(data_file, "r", encoding="utf-8") as f:
        programs = json.load(f)
    with open(embeddings_file, "rb") as f:
        embeddings_dict = pickle.load(f)
    
    query_embedding = generate_embedding(query)
    if not query_embedding:
        return "Error generating embedding for query."
    
    best_score = -1.0
    best_program = None
    for prog in programs:
        slug = prog.get("slug")
        emb = embeddings_dict.get(slug)
        if emb:
            score = cosine_similarity(query_embedding, emb)
            if score > best_score:
                best_score = score
                best_program = prog
                
    MIN_SIMILARITY_THRESHOLD = 0.5
    if best_score < MIN_SIMILARITY_THRESHOLD or best_program is None:
        return "No close match found for your query."
    
    prog_type = best_program.get("type", "unknown")
    title = best_program.get("title", "No Title Found")
    description = best_program.get("description", "")
    
    details = f"Program Details: {title}\nType: {prog_type}\n"
    
    if prog_type == "masters":
        if isinstance(description, dict) and "content" in description:
            plain_text = ""
            for block in description.get("content", []):
                for child in block.get("content", []):
                    plain_text += child.get("value", "") + " "
            description_text = plain_text.strip()
        else:
            description_text = str(description)
        details += f"Description: {description_text}\n"
    else:
        details += f"Description: {description}\n"
        if best_program.get("studyFee") is not None:
            details += f"Study Fee: {best_program.get('studyFee')}\n"
        if best_program.get("duration"):
            details += f"Duration: {best_program.get('duration')}\n"
        if best_program.get("modules"):
            details += "Modules:\n"
            for i, mod in enumerate(best_program.get("modules"), start=1):
                details += f"  {i}. {mod}\n"
    
    return details.strip()

def recommend_tool(query: str) -> str:
    """
    Provides program recommendations based on the user's background using embeddings
    for a more robust analysis of the background text.
    
    The tool will:
      - Generate an embedding for the user's background.
      - Compute similarity scores for each program in both the "masters" and "executive_master" categories.
      - Recommend programs from the category that best aligns with the background.
      - If the similarity scores are low across the board, advise the user accordingly.
    """
    background = query.strip()
    if not background:
        # If background is empty, default to recommending all masters.
        category = "masters"
    else:
        # Generate embedding for the background.
        bg_embedding = generate_embedding(background)
        if not bg_embedding:
            return "Error generating embedding for your background."

        # Load programs and embeddings.
        with open("data/programs.json", "r", encoding="utf-8") as f:
            programs = json.load(f)
        with open("data/program_embeddings.pkl", "rb") as f:
            embeddings = pickle.load(f)
        
        # Separate programs by category.
        masters = [p for p in programs if p.get("type") == "masters"]
        exec_masters = [p for p in programs if p.get("type") == "executive_master"]
        
        # Compute aggregate similarity scores for each category.
        def avg_similarity(progs):
            scores = []
            for prog in progs:
                slug = prog.get("slug")
                prog_emb = embeddings.get(slug)
                if prog_emb:
                    scores.append(cosine_similarity(bg_embedding, prog_emb))
            return sum(scores) / len(scores) if scores else 0

        masters_score = avg_similarity(masters)
        exec_score = avg_similarity(exec_masters)
        
        # Decide category based on which has a higher average similarity.
        if masters_score >= exec_score:
            category = "masters"
        else:
            category = "executive_master"

        if masters_score < 0.3 and exec_score < 0.3:
            return ("Our university primarily offers graduate programs, and it seems your background "
                    "does not strongly align with these. You might want to consider obtaining a bachelor's degree first.")
    
    # Filter programs based on the selected category.
    filtered = [p for p in programs if p.get("type") == category]
    if not filtered:
        return f"Sorry, no programs found in the '{category}' category."
    
    # Optionally, rank the filtered programs by embedding similarity.
    ranked = sorted(filtered, key=lambda p: cosine_similarity(
        bg_embedding, embeddings.get(p.get("slug"), [])), reverse=True) if background else filtered
    recommended_titles = [p.get("title") for p in ranked][:3]
    rec_list = "\n".join(f"{i+1}. {title}" for i, title in enumerate(recommended_titles))
    
    prompt = (
        f"You are a creative academic advisor. The user provided the following background: \"{background}\".\n"
        f"Based on this, recommend the following programs in the '{category}' category:\n"
        f"{rec_list}\n\n"
        "Please provide a creative recommendation message explaining why these programs might be an excellent fit for the user."
    )
    
    try:
        response = openai.Completion.create(
            model="gpt-4o",
            prompt=prompt,
            max_tokens=300,
            temperature=0.9
        )
        return response.choices[0].text.strip()
    except Exception as e:
        print(f"Error generating completion: {e}")
        return "An error occurred while generating the response."

# Set up the tools for the agent.
tools = [
    Tool(
        name="GetAllMasters",
        func=get_all_masters_tool,
        description="Returns a grouped list of all university masters programs."
    ),
    Tool(
        name="GetAllExecutiveMasters",
        func=get_all_executive_masters_tool,
        description="Returns a grouped list of all university executive masters programs."
    ),
    Tool(
        name="GetProgramDetails",
        func=get_program_details_tool,
        description="Given a specific program, returns detailed information about the best matching program using embedding-based matching."
    ),
    Tool(
        name="RecommendPrograms",
        func=recommend_tool,
        description="Provides creative program recommendations based on the user's background."
    )
]

llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)

agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True
)

if __name__ == "__main__":
    # Example interactions:
    print("=== Get All Programs ===")
    result_all = agent.run("List all programs available at the university.")
    print(result_all)
    
    print("\n=== Get Program Details ===")
    # Even with typos, the embedding matching should find the correct program.
    result_details = agent.run("I want details regarding the Artfcial Intlgc program")
    print(result_details)
    
    print("\n=== Recommend Programs ===")
    result_recommend = agent.run("I have a Bachelor's in Business and want to advance in finance.")
    print(result_recommend)