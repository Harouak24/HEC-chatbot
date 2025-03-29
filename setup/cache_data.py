import json
import os

from .contentful_client import (
    get_masters_programs,
    get_executive_masters,
    get_executive_certificates,
    get_executive_bachelors,
    get_master_certificates
)

def generate_local_cache(output_file: str = "data/programs.json"):
    """
    Fetches program data from Contentful (Masters, Executive Masters, Executive Certificates,
    Executive Bachelors, and Master Certificates) and consolidates them into a single JSON file.
    """
    # 1. Fetch data from Contentful
    masters = get_masters_programs()
    exec_masters = get_executive_masters()
    exec_certs = get_executive_certificates("")  # Fetch all executive certificates
    exec_bachelors = get_executive_bachelors()
    master_certs = get_master_certificates()

    # 2. Transform each list into a uniform structure

    # --- Masters ---
    masters_list = [
        {
            "type": "masters",
            "title": item.get("title", ""),
            "slug": item.get("slug", ""),
            "description": item.get("description", {}).get("json", {}),
            "studyFee": None,
            "applicationFee": None,
            "registrationFee": None,
            "duration": None,
            "modules": []
        }
        for item in masters
    ]

    # --- Executive Masters ---
    exec_masters_list = [
        {
            "type": "executive_master",
            "title": item.get("title", ""),
            "slug": item.get("slug", ""),
            "description": item.get("description", ""),
            "studyFee": item.get("studyFee", ""),
            "applicationFee": item.get("applicationFee", ""),
            "registrationFee": item.get("registrationFee", ""),
            "duration": None,
            "modules": [m.get("title", "") for m in item.get("modulesCollection", {}).get("items", [])]
        }
        for item in exec_masters
    ]

    # --- Executive Certificates ---
    exec_certs_list = [
        {
            "type": "executive_certificate",
            "title": item.get("title", ""),
            "slug": item.get("slug", ""),
            "description": item.get("description", ""),
            "studyFee": item.get("studyFee", ""),
            "applicationFee": item.get("applicationFee", ""),
            "registrationFee": item.get("registrationFee", ""),
            "duration": item.get("duration", ""),
            "modules": []
        }
        for item in exec_certs
    ]

    # --- Executive Bachelors ---
    exec_bachelors_list = [
        {
            "type": "executive_bachelor",
            "title": item.get("title", ""),
            "slug": item.get("slug", ""),
            "description": item.get("description", ""),
            "studyFee": item.get("studyFee", ""),
            "applicationFee": item.get("applicationFee", ""),
            "registrationFee": item.get("registrationFee", ""),
            "duration": None,
            "modules": []
        }
        for item in exec_bachelors
    ]

    # --- Master Certificates ---
    master_certs_list = [
        {
            "type": "master_certificate",
            "title": item.get("title", ""),
            "slug": None,
            "description": None,
            "studyFee": None,
            "applicationFee": None,
            "registrationFee": None,
            "duration": None,
            "modules": []
        }
        for item in master_certs
    ]

    # 3. Combine all data into a single list
    combined_programs = (
        masters_list + exec_masters_list + exec_certs_list + exec_bachelors_list + master_certs_list
    )

    # 4. Save to JSON
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(combined_programs, f, indent=2, ensure_ascii=False)
    
    print(f"Local cache generated at: {output_file}")

if __name__ == "__main__":
    generate_local_cache()