import os
import json
import pickle
import numpy as np
from agents import Agent, Runner, function_tool
from langchain_openai import OpenAIEmbeddings
import gradio as gr
from dotenv import load_dotenv
import asyncio

load_dotenv()

# Ensure OpenAI API key is set
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

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
    Generates an embedding for the given text using the OpenAI Text Embedding API.
    """
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        embedding = embeddings.embed_query(text)
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return []

@function_tool
def get_all_masters() -> str:
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
    output = f"HEC Masters:\n"
    for i, m in enumerate(masters, start=1):
        output += f"  {i}. {m}\n"
    output += "\n"
    return output.strip()

@function_tool
def get_all_executive_masters() -> str:
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
    output = f"HEC Executive Masters:\n"
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

@function_tool
def get_all_executive_bachelors() -> str:
    """
    Returns a grouped list of all university executive bachelor programs.
    """
    with open("data/programs.json", "r", encoding="utf-8") as f:
        programs = json.load(f)
    bachelors = []
    for prog in programs:
        if prog.get("type") == "executive_bachelor":
            bachelors.append(prog.get("title"))
    output = "HEC Executive Bachelors:\n"
    for i, title in enumerate(bachelors, start=1):
        output += f"  {i}. {title}\n"
    return output.strip()

@function_tool
def get_all_master_certificates() -> str:
    """
    Returns a grouped list of all master certificate programs.
    """
    with open("data/programs.json", "r", encoding="utf-8") as f:
        programs = json.load(f)
    certificates = []
    for prog in programs:
        if prog.get("type") == "master_certificate":
            certificates.append(prog.get("title"))
    output = "HEC Master Certificates:\n"
    for i, title in enumerate(certificates, start=1):
        output += f"  {i}. {title}\n"
    return output.strip()

@function_tool
def get_program_details(query: str) -> str:
    """
    Uses embedding matching to find the best program for a given query,
    then returns its detailed information.
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
                
    MIN_SIMILARITY_THRESHOLD = 0.25
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

@function_tool
def recommend_programs(background: str, interest: str) -> str:
    """
    Provides program recommendations based on the user's background and interest.
    """
    # Handle empty strings as if no input was provided
    if not background:
        background = ""
    if not interest:
        interest = ""
        
    combined_query = (background + " " + interest).strip()
    data_file = "data/programs.json"
    embeddings_file = "data/program_embeddings.pkl"
    
    if not os.path.exists(data_file) or not os.path.exists(embeddings_file):
        return json.dumps({"error": "Program data or embeddings not found."})
    
    with open(data_file, "r", encoding="utf-8") as f:
        programs = json.load(f)
    with open(embeddings_file, "rb") as f:
        embeddings = pickle.load(f)
    
    if not combined_query:
        category = "masters"
        bg_embedding = None
    else:
        bg_embedding = generate_embedding(combined_query)
        if not bg_embedding:
            return json.dumps({"error": "Error generating embedding for your input."})
        
        masters = [p for p in programs if p.get("type") == "masters"]
        exec_masters = [p for p in programs if p.get("type") == "executive_master"]
        
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
        
        if masters_score < 0.3 and exec_score < 0.3:
            return json.dumps({
                "error": ("Your background and interests do not strongly align with our graduate programs. "
                          "Please consider obtaining a bachelor's degree first.")
            })
        
        category = "masters" if masters_score >= exec_score else "executive_master"
    
    filtered = [p for p in programs if p.get("type") == category]
    if not filtered:
        return json.dumps({"error": f"Sorry, no programs found in the '{category}' category."})
    
    if bg_embedding:
        ranked = sorted(
            filtered,
            key=lambda p: cosine_similarity(bg_embedding, embeddings.get(p.get("slug"), [])),
            reverse=True
        )
    else:
        ranked = filtered
    recommended_titles = [p.get("title") for p in ranked][:3]
    
    result = {
        "category": category,
        "recommended_programs": recommended_titles
    }
    return json.dumps(result)

@function_tool
def rag_search(query: str) -> str:
    """
    Retrieves relevant content from the cached HEC website data.
    """
    data_file = "data/hec_webpages.json"
    embeddings_file = "data/hec_webpages_embeddings.pkl"
    
    if not os.path.exists(data_file) or not os.path.exists(embeddings_file):
        return json.dumps({"error": "Cached website data or embeddings not found."})
    
    with open(data_file, "r", encoding="utf-8") as f:
        docs = json.load(f)
    with open(embeddings_file, "rb") as f:
        embeddings_dict = pickle.load(f)
    
    query_emb = generate_embedding(query)
    if not query_emb:
        return json.dumps({"error": "Error generating embedding for your query."})
    
    similarities = []
    for i, doc in enumerate(docs):
        text = doc.get("page_content", "")
        key = f"doc_{i}"
        doc_emb = embeddings_dict.get(key)
        if doc_emb:
            sim = cosine_similarity(query_emb, doc_emb)
            similarities.append((sim, text))
    
    similarities.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [text for sim, text in similarities[:5]]
    concatenated_context = "\n\n".join(top_chunks)
    
    result = {
        "retrieved_chunks": top_chunks,
        "context": concatenated_context,
        "num_chunks": len(top_chunks)
    }
    return json.dumps(result)

# Create the academic advisor agent
academic_advisor = Agent(
    name="HEC Academic Advisor",
    instructions="""You are an academic advisor for HEC Rabat. HEC University is a graduate school that offers masters programs as well as executive masters programs. 
    You have access to a list of all programs available at the university. You can provide detailed information about a specific program, recommend programs based on a user's background and interest, and more. 
    Note: All responses should be strictly related to HEC Rabat. Please do not reference or confuse HEC Rabat with HEC Paris or any other HEC institutions.
    
    You have access to the following tools:
    - get_all_masters(): Lists all master's programs
    - get_all_executive_masters(): Lists all executive master's programs
    - get_program_details(query): Gets detailed information about a specific program
    - get_all_executive_bachelors(): Lists all executive bachelor programs
    - get_all_master_certificates(): Lists all master certificate programs
    - recommend_programs(background, interest): Recommends programs based on background and interests
    - rag_search(query): Searches the HEC website for relevant information
    
    Use these tools to provide accurate and helpful information to students.""",
    tools=[get_all_masters, get_all_executive_masters, get_all_executive_bachelors, get_all_master_certificates, get_program_details, recommend_programs, rag_search],
    model="gpt-4o-mini"
)

async def process_query(query: str) -> str:
    """
    Process a user query using the academic advisor agent asynchronously.
    """
    result = await Runner.run(academic_advisor, query)
    return result.final_output

# Create the Gradio interface
iface = gr.Interface(
    fn=process_query,
    inputs=gr.Textbox(
        label="Enter your question",
        placeholder="Type your query here..."
    ),
    outputs=gr.Markdown(label="Advisor Response"),
    title="HEC University Academic Advisor",
    description="Ask questions about HEC University masters and executive masters programs.",
    allow_flagging=False
)

if __name__ == "_main_":
    iface.queue().launch()