# Prueba para probar el código de la librería allosteric_analyzer

import requests
from Bio import SeqIO
from io import StringIO

uniprot_id = "P07550"  # Código de la proteína en UniProt
fasta_url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
response = requests.get(fasta_url)

if response.status_code == 200:
    fasta_content = response.text
    fasta_io = StringIO(fasta_content)
    record = SeqIO.read(fasta_io, "fasta")
    
    # Obtener la secuencia de aminoácidos
    prot1 = record.seq
    
    # Imprimir solo la secuencia de aminoácidos
    print(prot1)
else:
    print("Error al obtener la secuencia")

from allosteric_analyzer import AllosticHeadAnalyzer

# Definir los sitios alostéricos
# Los sitios descritos en ASD para esta proteína son: C118, S193, T205, L379, N418, V83, M117, Y199, L414
allosteric_sites = [118, 193, 205, 379, 83, 117, 199, 414]

# Crear un objeto de la clase AllostericHeadAnalyzer
analyzer = AllosticHeadAnalyzer()
results = analyzer.analyze_protein(prot1, allosteric_sites)

# Imprimir los resultados
impact_scores = results["impacts"].squeeze().tolist()
snrs = results["snrs"].squeeze().tolist()

print(len(impact_scores))