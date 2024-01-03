import requests
from bs4 import BeautifulSoup

# List of CID codes extracted from the links
cid_list = [...]  # Replace with your actual list of CID codes

# Template URL for constructing the actual URLs
template_url = "https://pubchem.ncbi.nlm.nih.gov/compound/{CID}#section=Taxonomy&fullscreen=true"

# Loop through the list of CID codes
for cid in cid_list:
    # Construct the actual URL for the current CID code
    url = template_url.replace("{CID}", str(cid))
    
    # Send GET request to the URL and retrieve HTML content
    response = requests.get(url)
    html_content = response.text
    
    # Parse the HTML content
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Extract the table data from the parsed HTML
    # (Specific steps for parsing will depend on HTML structure)
    table = soup.find('table')
    # Extract table data and save it to desired format
    
    # Repeat for all CID codes in the list
