from app.classification import classify_intent
from app.general_inquiries import handle_general_inquiries
from app.program_details import handle_program_details
from app.recommendations import recommend_programs

def chatbot_orchestrator(user_query: str) -> str:
    """
    Central orchestrator function for the university chatbot.
    
    Args:
        user_query (str): The user's input query.
    
    Returns:
        str: The response from the appropriate chain.
    """
    classification = classify_intent(user_query)
    intent = classification.get("intent", "general_inquiry")
    program_name = classification.get("programName", "").strip()
    background = classification.get("background", "").strip()
    
    # Route the request based on the classified intent.
    if intent == "general_inquiry":
        return handle_general_inquiries()
    elif intent == "program_specific":
        if not program_name:
            return "I couldn't identify which program you are asking about. Please specify the program name."
        return handle_program_details(program_name)
    elif intent == "recommendation":
        # If background is empty, recommend all masters by default.
        return recommend_programs(background)
    else:
        return "I'm sorry, I didn't understand your request."

# Optional test code for local testing
if __name__ == "__main__":
    user_input = input("Enter your query: ")
    response = chatbot_orchestrator(user_input)
    print("\nChatbot response:\n", response)