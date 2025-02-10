import os
import requests
from dotenv import load_dotenv

load_dotenv()

# Contentful Configuration
CONTENTFUL_SPACE_ID = os.getenv("CONTENTFUL_SPACE_ID")
CONTENTFUL_AUTH_TOKEN = os.getenv("CONTENTFUL_AUTH_TOKEN")
CONTENTFUL_API_URL = f"https://graphql.contentful.com/content/v1/spaces/{CONTENTFUL_SPACE_ID}"


def query_contentful(query: str, variables: dict = None) -> dict:
    """
    Sends a POST request to the Contentful GraphQL API with the given query and variables.

    Args:
        query (str): The GraphQL query string.
        variables (dict, optional): Any variables referenced by the GraphQL query.

    Returns:
        dict: The JSON response from Contentful if successful.

    Raises:
        RuntimeError: If the response status code is not 200.
    """
    headers = {
        "Authorization": CONTENTFUL_AUTH_TOKEN,
        "Content-Type": "application/json"
    }

    response = requests.post(
        CONTENTFUL_API_URL, 
        headers=headers, 
        json={"query": query, "variables": variables}
    )

    if response.status_code != 200:
        error_msg = (
            f"Contentful query failed with status {response.status_code}.\n"
            f"Response: {response.text}"
        )
        raise RuntimeError(error_msg)

    return response.json()

def get_masters_programs() -> list:
    """
    Fetches all Master's programs from Contentful.

    Returns:
        list: A list of dictionaries representing Master's programs. Each item
              contains 'title', 'slug', and 'description' fields.
    """
    query = """
    query {
      hecPgeMastersCollection {
        items {
          title
          slug
          description {
            json
          }
        }
      }
    }
    """

    response_data = query_contentful(query)
    data = response_data.get("data", {})
    masters_collection = data.get("hecPgeMastersCollection", {})
    items = masters_collection.get("items", [])
    return items

def get_executive_masters() -> list:
    """
    Fetches all Executive Master's programs from Contentful.

    Returns:
        list: A list of dictionaries representing Executive Master's programs. Each item
              contains fields like 'title', 'slug', 'description', 'studyFee', 'applicationFee',
              'registrationFee', and 'modulesCollection'.
    """
    query = """
    query {
      hecPgeExecutiveMastersCollection {
        items {
          title
          slug
          description
          studyFee
          applicationFee
          registrationFee
          modulesCollection {
            items {
              title
            }
          }
        }
      }
    }
    """

    response_data = query_contentful(query)
    data = response_data.get("data", {})
    exec_masters_collection = data.get("hecPgeExecutiveMastersCollection", {})
    items = exec_masters_collection.get("items", [])
    return items

def get_executive_certificates(title_filter: str = "") -> list:
    """
    Fetches Executive Certificates from Contentful, optionally filtering by a substring of the title.

    If no title_filter is provided, it returns all Executive Certificates.

    Args:
        title_filter (str, optional): A substring to filter titles by. Defaults to "" (empty string).

    Returns:
        list: A list of dictionaries representing Executive Certificates. Each item
              contains fields like 'title', 'slug', 'description', 'studyFee',
              'applicationFee', 'registrationFee', and 'duration'.
    """
    query = """
    query($title: String!) {
      hecPgeExecutiveCertificatesCollection(where: { title_contains: $title }) {
        items {
          title
          slug
          description
          studyFee
          applicationFee
          registrationFee
          duration
        }
      }
    }
    """
    # If title_filter is empty, it will return all certificates because `title_contains: ""` matches all.
    variables = {"title": title_filter}

    response_data = query_contentful(query, variables)
    data = response_data.get("data", {})
    exec_certs_collection = data.get("hecPgeExecutiveCertificatesCollection", {})
    items = exec_certs_collection.get("items", [])
    return items