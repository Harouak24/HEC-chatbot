import json
import os

from services.contentful_client import (
    get_masters_programs,
    get_executive_masters,
    get_executive_certificates
)

def generate_local_cache(output_file: str = "data/programs.json"):
    """
    Fetches program data from Contentful (masters, executive masters, executive certificates)
    and consolidates them into a single JSON file to reduce repeated queries.

    Args:
        output_file (str): The path where the combined JSON file will be saved.
    """
    # 1. Fetch data from Contentful
    masters = get_masters_programs()
    exec_masters = get_executive_masters()
    exec_certs = get_executive_certificates("")  # Passing empty string to get all certs

    # 2. Transform each list into a uniform structure

    # --- Masters ---
    masters_list = []
    for item in masters:
        # 'description' is a rich-text JSON object under item["description"]["json"]
        description_json = item.get("description", {}).get("json", {})
        masters_list.append({
            "type": "masters",
            "title": item.get("title", ""),
            "slug": item.get("slug", ""),
            "description": description_json,  # or convert to string if you prefer
            "studyFee": None,        # Not defined for Masters
            "applicationFee": None,  # Not defined for Masters
            "registrationFee": None, # Not defined for Masters
            "duration": None,        # Not defined for Masters
            "modules": []            # Not defined for Masters
        })

    # --- Executive Masters ---
    exec_masters_list = []
    for item in exec_masters:
        modules = []
        if "modulesCollection" in item and item["modulesCollection"].get("items"):
            modules = [m.get("title", "") for m in item["modulesCollection"]["items"]]

        exec_masters_list.append({
            "type": "executive_master",
            "title": item.get("title", ""),
            "slug": item.get("slug", ""),
            "description": item.get("description", ""),
            "studyFee": item.get("studyFee", ""),
            "applicationFee": item.get("applicationFee", ""),
            "registrationFee": item.get("registrationFee", ""),
            "duration": None,  # Typically not defined for Exec Masters
            "modules": modules
        })

    # --- Executive Certificates ---
    exec_certs_list = []
    for item in exec_certs:
        exec_certs_list.append({
            "type": "executive_certificate",
            "title": item.get("title", ""),
            "slug": item.get("slug", ""),
            "description": item.get("description", ""),
            "studyFee": item.get("studyFee", ""),
            "applicationFee": item.get("applicationFee", ""),
            "registrationFee": item.get("registrationFee", ""),
            "duration": item.get("duration", ""),
            "modules": []  # Not defined for Exec Certificates
        })

    # 3. Combine them into a single list
    combined_programs = masters_list + exec_masters_list + exec_certs_list

    # 4. Save to JSON
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(combined_programs, f, indent=2, ensure_ascii=False)
    
    print(f"Local cache generated at: {output_file}")

if __name__ == "__main__":
    # Run this script directly to generate a fresh programs.json
    generate_local_cache()