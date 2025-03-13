# I'll run a test the code.

import requests

uniprot_id = "O15530"  # Código de la proteína en UniProt
url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"

response = requests.get(url)

if response.status_code == 200:
    print(response.text)  # Imprime la secuencia en formato FASTA
else:
    print("Error al obtener la secuencia")

# from allosteric_analyzer import AllosticHeadAnalyzer