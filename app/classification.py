import json
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.schema import AIMessage
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 1. Prompt: Provide detailed instructions and sample JSON outputs.
#    We want the model to respond ONLY with well-formed JSON containing:
#    - intent (general_inquiry | program_specific | recommendation)
#    - programName
#    - background
CLASSIFICATION_SYSTEM_PROMPT = """
You are an intent classification system for a university chatbot.

We have three main intents:
1. "general_inquiry": The user wants a general list or overview of programs.
2. "program_specific": The user wants details about a specific program.
3. "recommendation": The user wants you to suggest a program based on their background/interests.

Additionally:
- If the user mentions a program (e.g., "Executive Master in Finance"), capture it in "programName".
- If the user is asking for a recommendation, capture any background info they provide in "background".
- If they do not specify a program or background, use an empty string.

Return your answer as valid JSON with exactly these keys: 
{{
  "intent": "...",
  "programName": "...",
  "background": "..."
}}

Examples:

Example 1:
User's message: "Hello, can you show me the list of programs you have?"
JSON Output:
{{
  "intent": "general_inquiry",
  "programName": "",
  "background": ""
}}

Example 2:
User's message: "I want details about the Executive Master in Finance"
JSON Output:
{{
  "intent": "program_specific",
  "programName": "Executive Master in Finance",
  "background": ""
}}

Example 3:
User's message: "I have a background in engineering and want a short management program"
JSON Output:
{{
  "intent": "recommendation",
  "programName": "",
  "background": "engineering background, short management program"
}}

IMPORTANT: Do not include any extra text or commentary—only JSON.

Now, analyze the user's message and produce your JSON:
"""

CLASSIFICATION_HUMAN_TEMPLATE = """\
User's message: {user_query}
"""

def build_classification_prompt() -> ChatPromptTemplate:
    """
    Constructs a ChatPromptTemplate with separate system & human messages.
    """
    system_message = SystemMessagePromptTemplate.from_template(CLASSIFICATION_SYSTEM_PROMPT)
    human_message = HumanMessagePromptTemplate.from_template(CLASSIFICATION_HUMAN_TEMPLATE)

    # Return a ChatPromptTemplate that concatenates system + human messages
    return ChatPromptTemplate.from_messages([system_message, human_message])

# Create a ChatPromptTemplate
classification_prompt = build_classification_prompt()

# 3. Instantiate an LLM (OpenAI). 
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    openai_api_key=OPENAI_API_KEY,
    temperature=0.1,
    max_tokens=200,
    max_retries=2
)

# 4. Create a "runnable" pipeline by piping prompt into the LLM.
classification_runnable = classification_prompt | llm

def classify_intent(user_query: str) -> dict:
    """
    Classifies the user's intent using LangChain's LLMChain.
    
    Returns:
        dict: A dictionary with keys "intent", "programName", and "background".
              e.g. {
                  "intent": "program_specific",
                  "programName": "Executive Master in Finance",
                  "background": ""
              }
    """
    # 1. Invoke the pipeline (prompt + LLM) with a dict input.
    result = classification_runnable.invoke({"user_query": user_query})

    if isinstance(result, AIMessage):
        response_str = result.content  # The string output from the chat model
    else:
        # In some versions, it might just return a str. We'll handle both.
        response_str = str(result)
    
    # 2. Try to parse the model's output as JSON
    try:
        data = json.loads(response_str.strip())
    except json.JSONDecodeError:
        # Fallback if parsing fails
        return {
            "intent": "general_inquiry",
            "programName": "",
            "background": ""
        }

    # 3. Ensure the keys exist
    for key in ["intent", "programName", "background"]:
        if key not in data:
            data[key] = ""
    
    return data

# 5. Optional: Test script
if __name__ == "__main__":
    test_query = "I am a highschooler interested in computer science"
    result = classify_intent(test_query)
    print("Classification Result:", result)