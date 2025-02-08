from services.contentful_client import (
    get_masters_programs, 
    get_executive_masters, 
    get_executive_certificates
)

def test_contentful_api():
    masters = get_masters_programs()
    print("Masters:", masters)

    exec_masters = get_executive_masters()
    print("Executive Masters:", exec_masters)

    # No filter => fetch all
    exec_certs = get_executive_certificates()
    print("Executive Certificates:", exec_certs)

if __name__ == "__main__":
    test_contentful_api()