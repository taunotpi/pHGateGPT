import os
import requests

# Base directory for data downloads
BASE_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# -------------------------------------
# KEGG Downloads
# -------------------------------------
def download_kegg_data():
    # Define URLs and output filenames for KEGG entries.
    kegg_files = {
        "K02144_vatpase.txt": "https://www.genome.jp/dbget-bin/www_bget?ko:K02144",
        "K01365_cathepsin_L.txt": "https://www.genome.jp/dbget-bin/www_bget?hsa:CTSL",
        "K14725_NHE9.txt": "https://www.genome.jp/dbget-bin/www_bget?ko:K14725"
    }

    for filename, url in kegg_files.items():
        output_path = os.path.join(BASE_OUTPUT_DIR, filename)
        try:
            print(f"Downloading KEGG data from {url} ...")
            response = requests.get(url)
            if response.status_code == 200:
                with open(output_path, "w") as f:
                    f.write(response.text)
                print(f"Saved KEGG data to {output_path}")
                if os.path.getsize(output_path) == 0:
                    print(f"Warning: {output_path} is empty!")
            else:
                print(f"Failed to download {url}. HTTP status: {response.status_code}")
        except Exception as e:
            print(f"Error downloading KEGG data from {url}: {e}")

# -------------------------------------
# UniProt Downloads
# -------------------------------------
def download_uniprot_data():
    # Define URLs and output filenames for UniProt entries.
    uniprot_files = {
        "vatpase_subunit_H_Q9UI12.html": "https://www.uniprot.org/uniprotkb/Q9UI12/entry",
        "vatpase_subunit_A_P38606.html": "https://www.uniprot.org/uniprotkb/P38606/entry",
        "cathepsin_L_Q5T8F0.html": "https://www.uniprot.org/uniprotkb/Q5T8F0/entry",
        "cathepsin_L_Q5K630.html": "https://www.uniprot.org/uniprotkb/Q5K630/entry",
        "NHE9_Q8IVB4.html": "https://www.uniprot.org/uniprotkb/Q8IVB4/entry"
    }

    for filename, url in uniprot_files.items():
        output_path = os.path.join(BASE_OUTPUT_DIR, filename)
        try:
            print(f"Downloading UniProt data from {url} ...")
            response = requests.get(url)
            if response.status_code == 200:
                with open(output_path, "w") as f:
                    f.write(response.text)
                print(f"Saved UniProt data to {output_path}")
                if os.path.getsize(output_path) == 0:
                    print(f"Warning: {output_path} is empty!")
            else:
                print(f"Failed to download {url}. HTTP status: {response.status_code}")
        except Exception as e:
            print(f"Error downloading UniProt data from {url}: {e}")

# -------------------------------------
# NCBI GEO / Gene Downloads
# -------------------------------------
def download_ncbi_geo_data():
    # Define URLs and output filenames for NCBI gene pages.
    ncbi_files = {
        "gene_535.html": "https://www.ncbi.nlm.nih.gov/gene/535",
        "gene_525.html": "https://www.ncbi.nlm.nih.gov/gene/525",
        "gene_529.html": "https://www.ncbi.nlm.nih.gov/gene/529",
        "cathepsin_L_1514.html": "https://www.ncbi.nlm.nih.gov/gene/1514",
        "cathepsin_L_1514_details.html": "https://www.ncbi.nlm.nih.gov/gene?Db=gene&Cmd=DetailsSearch&Term=1514",
        "gene_285195.html": "https://www.ncbi.nlm.nih.gov/gene/285195"
    }

    for filename, url in ncbi_files.items():
        output_path = os.path.join(BASE_OUTPUT_DIR, filename)
        try:
            print(f"Downloading NCBI data from {url} ...")
            response = requests.get(url)
            if response.status_code == 200:
                with open(output_path, "w") as f:
                    f.write(response.text)
                print(f"Saved NCBI data to {output_path}")
                if os.path.getsize(output_path) == 0:
                    print(f"Warning: {output_path} is empty!")
            else:
                print(f"Failed to download {url}. HTTP status: {response.status_code}")
        except Exception as e:
            print(f"Error downloading NCBI data from {url}: {e}")

# -------------------------------------
# Main Function
# -------------------------------------
def main():
    download_kegg_data()
    download_uniprot_data()
    download_ncbi_geo_data()

if __name__ == "__main__":
    main()
