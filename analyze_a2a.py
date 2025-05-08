"""
Adenosine A2A Receptor Allosteric Site Analysis
=============================================

This script analyzes the allosteric sites of the human Adenosine A2A receptor
using the ESM2 protein language model attention patterns. It identifies which
attention heads are most sensitive to known allosteric sites.

The analysis focuses on four validated allosteric sites:
- Position 85: Part of the sodium binding pocket
- Position 89: Part of the sodium binding pocket
- Position 246: Key residue in the activation mechanism
- Position 253: Involved in conformational changes

The script uses the AllosticHeadAnalyzer class to:
1. Process the protein sequence through ESM2
2. Analyze attention patterns around allosteric sites
3. Calculate sensitivity scores for each attention head
4. Identify heads that are most sensitive to allosteric communication

Usage
-----
$ python analyze_a2a.py

Output
------
The script prints:
- Sensitivity scores for each of the 20 attention heads
- List of heads with above-average sensitivity to allosteric sites

Example Output
-------------
Allosteric sensitivity scores per attention head:
Head 0: 0.048
Head 1: 0.040
...
Head 19: 0.050

Most sensitive heads to allosteric sites [85, 89, 246, 253]:
Head indices: [0, 2, 3, 5, 6, 7, 9, 10, 12, 15, 18, 19]

Notes
-----
The protein sequence is from PDB structure 3VG9, representing the
inactive state of the human Adenosine A2A receptor.

References
----------
1. Liu et al. (2012). Science, 337(6091), 232-236.
   DOI: 10.1126/science.1219218
2. Dong et al. (2024). Allo-Allo: Data-efficient prediction of 
   allosteric sites. bioRxiv. DOI: pending

Dependencies
-----------
- allosteric_analyzer.py
- torch
- fair-esm

Author
------
Aurelio A. Moya-García
Date: February 18, 2025
"""

import os
import torch
from Bio.Data import IUPACData
from allosteric_analyzer import AllosticHeadAnalyzer

def main():
    # Adenosine A2A receptor 3VG9
    sequence = "MPIMGSSVYITVELAIAVLAILGNVLVCWAVWLNSNLQNVTNYFVVSAAAADILVGVLAIPFAIAISTGFCAACHGCLFIACFVLVLTASSIFSLLAIAIDRYIAIRIPLRYNGLVTGTRAKGIIAICWVLSFAIGLTPMLGWNNCGQPKEGKAHSQGCGEGQVACLFEDVVPMNYMVYFNFFACVLVPLLLMLGVYLRIFLAARRQLKQMESQPLPGERARSTLQKEVHAAKSLAIIVGLFALCWLPLHIINCFTFFCPDCSHAPLWLMYLAIVLSHTNSVVNPFIYAYRIREFRQTFRKIIRSHVLRQQEPFKAAAAENLYFQ"
    allosteric_sites = [168, 169, 253, 277, 278]
    orthosteric_sites = [102, 110, 227, 231, 235] # Orthosteric sites for analysis predicted with a Protein Contact Network.
    pathway_sites = [55] # Pathway sites for analysis predicted with a Protein Contact Network.
    pathway_sites_3l = []
    one_to_three = IUPACData.protein_letters_1to3
    for i in pathway_sites:
        pathway_sites_3l.append(one_to_three[f"{sequence[i-1]}"])

    print("=" * 50)
    print(f"The allosteric pathway residues are: ")
    for idx, site in enumerate(pathway_sites):
        print(pathway_sites_3l[idx],site)
    
    # Initialize analyzer
    analyzer = AllosticHeadAnalyzer(threshold=0.3)
    
    # Basic analysis
    results = analyzer.analyze_protein(sequence, allosteric_sites)
    
    # Get both impact scores and SNR values
    impact_scores = results["impacts"].squeeze().tolist()
    snr_values = results["snrs"].squeeze().tolist()
    
    print("\nAllosteric sensitivity analysis per attention head:")
    print("Head | Impact Score | SNR")
    print("-" * 35)
    for head_idx in range(len(impact_scores)):
        print(f"{head_idx:4d} | {impact_scores[head_idx]:11.3f} | {snr_values[head_idx]:6.2f}")
    
    # Calculate means
    mean_impact = sum(impact_scores) / len(impact_scores)
    mean_snr = sum(snr_values) / len(snr_values)
    
    # Identify heads that are significant by both metrics
    sensitive_heads = [i for i, (impact, snr) in enumerate(zip(impact_scores, snr_values)) 
                      if impact > mean_impact and snr > 2.0]  # SNR > 2 is statistically significant
    
    print(f"\nMost sensitive heads to allosteric sites {allosteric_sites}:")
    print(f"(Impact > {mean_impact:.3f} and SNR > 2.0)")
    print(f"Head indices: {sensitive_heads}")

    # print("\nAnalyzing attention patterns for sensitive heads...")
    
    # Get attention maps
    attention_maps = analyzer.get_attention_maps(sequence)
    
    # Visualize and analyze each sensitive head
    for head_idx in sensitive_heads:
        analyzer.visualize_head_attention(
                attention_maps=attention_maps,
                allosteric_sites=allosteric_sites,
                orthosteric_sites=orthosteric_sites,
                pathway_sites=pathway_sites,
                head_idx=head_idx,
                sequence=sequence
)
        analyzer.analyze_head_connections(head_idx, attention_maps, sequence)
# 
#   Save attention maps for sensitive heads
    output_dir = "attention_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save sensitive head indices and their scores
    head_info = {
        'head_indices': sensitive_heads,
        'impact_scores': [impact_scores[i] for i in sensitive_heads],
        'snr_values': [snr_values[i] for i in sensitive_heads]
    }
    torch.save(head_info, os.path.join(output_dir, 'sensitive_heads.pt'))
    
    # Save attention maps for sensitive heads
    sensitive_attention = attention_maps[0, 0, sensitive_heads]  # [num_sensitive_heads, seq_len, seq_len]
    torch.save(sensitive_attention, os.path.join(output_dir, 'sensitive_attention_maps.pt'))
    
    print(f"Saved sensitive head information and attention maps to {output_dir}/")

if __name__ == "__main__":
    main()

