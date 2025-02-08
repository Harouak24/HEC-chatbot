import os
import openai
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

def generate_embedding(text: str) -> list:
    """
    Generates an embedding vector for the given text using OpenAI's Embeddings API.
    
    This function is critical for the chatbot's recommendation feature, where
    semantic similarity is used to match user queries against program descriptions.

    Args:
        text (str): The text to embed.

    Returns:
        list: A 1D list of floats representing the embedding vector.

    Raises:
        openai.error.OpenAIError: If there is any issue calling the OpenAI API.
    """
    try:
        response = openai.Embedding.create(
            input=[text],
            model="text-embedding-ada-002"
        )
        embedding = response["data"][0]["embedding"]
        return embedding
    except openai.error.OpenAIError as e:
        print(f"Error generating embedding: {e}")
        # Return an empty list or handle the error as needed
        return []

def generate_completion(
    prompt: str,
    model: str = "text-davinci-003",
    max_tokens: int = 300,
    temperature: float = 0.0
) -> str:
    """
    Generates a text completion for the given prompt using OpenAI's Completion API.

    This can be used for tasks like:
      - Summarizing content
      - Classifying user inputs (if not using LangChain LLMChain)
      - Providing direct answers based on user prompts

    Args:
        prompt (str): The prompt text.
        model (str, optional): The OpenAI model to use. Defaults to "text-davinci-003".
        max_tokens (int, optional): The maximum tokens in the response. Defaults to 300.
        temperature (float, optional): Sampling temperature. Defaults to 0.0 for deterministic outputs.

    Returns:
        str: The text response from OpenAI.

    Raises:
        openai.error.OpenAIError: If there is any issue calling the OpenAI API.
    """
    try:
        response = openai.Completion.create(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        text = response["choices"][0]["text"].strip()
        return text
    except openai.error.OpenAIError as e:
        print(f"Error generating completion: {e}")
        return "An error occurred while generating the response."