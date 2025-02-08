import json
import os
from collections import defaultdict

def handle_general_inquiries(data_file: str = "data/programs.json") -> str:
    """
    Lists all available programs, grouped by type, by reading from the locally cached JSON file.
    
    Args:
        data_file (str): Path to the JSON file containing program data.

    Returns:
        str: A formatted string grouping the programs by type:
             - Masters
             - Executive Masters
             - Executive Certificates
    """
    if not os.path.exists(data_file):
        return "No cached program data found. Please run your cache_data script."

    # 1. Load all programs
    with open(data_file, "r", encoding="utf-8") as f:
        programs = json.load(f)

    if not programs:
        return "No programs available at this time."

    # 2. Group programs by type
    #    The 'type' field can be: "masters", "executive_master", or "executive_certificate"
    grouped = defaultdict(list)
    for prog in programs:
        prog_type = prog.get("type", "other")  # fallback if type is missing
        grouped[prog_type].append(prog)

    # 3. Build the output lines
    lines = []

    # Masters
    masters_list = grouped.get("masters", [])
    if masters_list:
        lines.append("• Masters Programs")
        for i, prog in enumerate(masters_list, start=1):
            title = prog.get("title", "Untitled Program")
            lines.append(f"  {i}. {title}")
        lines.append("")  # blank line after the section

    # Executive Masters
    exec_masters_list = grouped.get("executive_master", [])
    if exec_masters_list:
        lines.append("• Executive Masters")
        for i, prog in enumerate(exec_masters_list, start=1):
            title = prog.get("title", "Untitled Program")
            lines.append(f"  {i}. {title}")
        lines.append("")

    # Executive Certificates
    exec_certs_list = grouped.get("executive_certificate", [])
    if exec_certs_list:
        lines.append("• Executive Certificates")
        for i, prog in enumerate(exec_certs_list, start=1):
            title = prog.get("title", "Untitled Program")
            lines.append(f"  {i}. {title}")
        lines.append("")

    # 4. If none found in any category
    if not (masters_list or exec_masters_list or exec_certs_list):
        return "No programs found for any known category."

    # 5. Join all lines
    return "\n".join(lines)

# Optional test
if __name__ == "__main__":
    output = handle_general_inquiries()
    print(output)
