import os
import json
import pickle
import numpy as np
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

#llm = wrap_openai(ChatOpenAI(model="gpt-4o", temperature=0.5))
llm = ChatOpenAI(model="gpt-4o", temperature=0.5)

prompt_template = """
    You are an academic advisor at HEC University. HEC University is a graduate school that offers masters programs as well as executive masters programs. You have access to a list of all programs available at the university. You can provide detailed information about a specific program, recommend programs based on a user's background and interest, and more.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", prompt_template),
        MessagesPlaceholder(variable_name="history"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
        ("human", "{input}"),
    ]
)

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

@tool
def get_all_masters_tool() -> str:
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

@tool
def get_all_executive_masters_tool() -> str:
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

@tool
def get_all_executive_bachelors_tool() -> str:
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

@tool
def get_all_master_certificates_tool() -> str:
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

@tool
def get_program_details_tool(query: str) -> str:
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

@tool
def recommend_programs_tool(background: str = "", interest: str = "") -> str:
    """
    Provides program recommendations based on the user's background and interest.
    
    This tool returns structured data (as a JSON string) that includes:
      - "category": either "masters", "executive_master", "executive_bachelor", 
                    "executive_certificate", or "master_certificate"
      - "recommended_programs": a list of recommended program titles (up to three)
    """
    combined_query = (background + " " + interest).strip()
    data_file = "data/programs.json"
    embeddings_file = "data/program_embeddings.pkl"
    
    if not os.path.exists(data_file) or not os.path.exists(embeddings_file):
        return json.dumps({"error": "Program data or embeddings not found."})
    
    with open(data_file, "r", encoding="utf-8") as f:
        programs = json.load(f)
    with open(embeddings_file, "rb") as f:
        embeddings = pickle.load(f)
    
    # If no input is provided, default to master's category.
    if not combined_query:
        category = "masters"
        bg_embedding = None
    else:
        bg_embedding = generate_embedding(combined_query)
        if not bg_embedding:
            return json.dumps({"error": "Error generating embedding for your input."})
        
        # Separate programs into broader categories.
        # For recommendations, we group traditional masters and master certificates together,
        # and executive masters, executive bachelors, and executive certificates together.
        masters = [p for p in programs if p.get("type") in ("masters", "master_certificate")]
        exec_programs = [p for p in programs if p.get("type") in ("executive_master", "executive_bachelor", "executive_certificate")]
        
        def avg_similarity(progs):
            scores = []
            for prog in progs:
                slug = prog.get("slug")
                prog_emb = embeddings.get(slug)
                if prog_emb:
                    scores.append(cosine_similarity(bg_embedding, prog_emb))
            return sum(scores) / len(scores) if scores else 0
        
        masters_score = avg_similarity(masters)
        exec_score = avg_similarity(exec_programs)
        
        # If both similarity scores are low, advise the user accordingly.
        if masters_score < 0.3 and exec_score < 0.3:
            return json.dumps({
                "error": ("Your background and interests do not strongly align with our graduate programs. "
                          "Please consider obtaining a bachelor's degree first.")
            })
        
        # Select the category with the higher average similarity.
        category = "masters" if masters_score >= exec_score else "executive_master"
    
    # Filter programs by the selected category.
    if category == "masters":
        filtered = [p for p in programs if p.get("type") in ("masters", "master_certificate")]
    else:
        filtered = [p for p in programs if p.get("type") in ("executive_master", "executive_bachelor", "executive_certificate")]
    
    if not filtered:
        return json.dumps({"error": f"Sorry, no programs found in the '{category}' category."})
    
    # Rank the filtered programs by cosine similarity if an embedding was generated.
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

@tool
def rag_tool(query: str) -> str:
    """
    Retrieves relevant content from the cached HEC website data and returns structured data.
    
    The agent can then use or refine this output in its final response.
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
    
    # Compute cosine similarity for each document chunk.
    similarities = []
    for i, doc in enumerate(docs):
        text = doc.get("page_content", "")
        key = f"doc_{i}"
        doc_emb = embeddings_dict.get(key)
        if doc_emb:
            sim = cosine_similarity(query_emb, doc_emb)
            similarities.append((sim, text))
    
    # Sort chunks by similarity (highest first) and select the top 5.
    similarities.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [text for sim, text in similarities[:5]]
    concatenated_context = "\n\n".join(top_chunks)
    
    result = {
        "retrieved_chunks": top_chunks,
        "context": concatenated_context,
        "num_chunks": len(top_chunks)
    }
    return json.dumps(result)

tools = [get_all_executive_masters_tool, get_all_masters_tool, get_program_details_tool, recommend_programs_tool, rag_tool]

agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools)

def academic_advisor(query):
    response = agent_executor.invoke({
        "input": query,
        "history": [],
        "agent_scratchpad": []
    })
    return response.get("output", "No output available.")

iface = gr.Interface(
    fn=academic_advisor,
    inputs=gr.Textbox(
        label="Enter your question",
        placeholder="Type your query here..."
    ),
    outputs=gr.Textbox(label="Advisor Response"),
    title="HEC University Academic Advisor",
    description="Ask questions about HEC University masters and executive masters programs."
)

if __name__ == "__main__":
    iface.launch()