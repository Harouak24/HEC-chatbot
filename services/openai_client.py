import os
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

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
            encoding_format="float"  # If supported by your account/model
        )
        # 'response' is a typed object
        embedding = response.data[0].embedding
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return []


def generate_completion(
    prompt: str
) -> str:
    """
    Generates a text completion for the given prompt using OpenAI's Completion API.

    Args:
        prompt (str): The prompt text.
        model (str, optional): The OpenAI model to use (e.g., "text-davinci-003").
        max_tokens (int, optional): The maximum tokens in the response. Defaults to 300.
        temperature (float, optional): The sampling temperature. Defaults to 0.0 for deterministic outputs.

    Returns:
        str: The text response from OpenAI, or a short error message if there's an issue.
    """
    try:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=300,
            temperature=0.0
        )
        return response.choices[0].text.strip()
    except Exception as e:
        print(f"Error generating completion: {e}")
        return "An error occurred while generating the response."