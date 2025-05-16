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
import numpy as np
from Bio.Data import IUPACData
from allosteric_analyzer import AllosticHeadAnalyzer

def main():
    # beta-adrenergic receptor 2 (ADRB2) sequence
    sequence = "MGQPGNGSAFLLAPNGSHAPDHDVTQERDEVWVVGMGIVMSLIVLAIVFGNVLVITAIAKFERLQTVTNYFITSLACADLVMGLAVVPFGAAHILMKMWTFGNFWCEFWTSIDVLCVTASIETLCVIAVDRYFAITSPFKYQSLLTKNKARVIILMVWIVSGLTSFLPIQMHWYRATHQEAINCYANETCCDFFTNQAYAIASSIVSFYVPLVIMVFVYSRVFQEAKRQLQKIDKSEGRFHVQNLSQVEQDGRTGHGLRRSSKFCLKEHKALKTLGIIMGTFTLCWLPFFIVNIVHVIQDNLIRKEVYILLNWIGYVNSGFNPLIYCRSPDFRIAFQELLCLRRSSLKAYGNGYSSNGNTGEQSGYHVEQEKENKLLCEDLPGTEDFVGHQGTVPSDNIDSQGRNCSTNDSLL"
    allosteric_res =  [113, 312, 203, 204, 207, 293, 296, 308]
    orthosteric_res = [] 
    pathway_res = []
    allosteric_res_3l = []
    one_to_three = IUPACData.protein_letters_1to3
    for i in allosteric_res:
        allosteric_res_3l.append(one_to_three[f"{sequence[i-1]}"])

    print("=" * 50)
    print(f"The allosteric residues are: ")
    for idx, site in enumerate(allosteric_res):
        print(allosteric_res_3l[idx], site)

    # Initialize analyzer
    analyzer = AllosticHeadAnalyzer(threshold=0.3)

    # Basic analysis
    results = analyzer.analyze_protein(sequence, allosteric_res)

    impact_scores_tensor = results["impacts"]
    snr_values_tensor = results["snrs"]

    # Asegura que están en 2D
    assert impact_scores_tensor.ndim == 2, f"Impact tensor must be 2D, got {impact_scores_tensor.shape}"
    assert snr_values_tensor.ndim == 2, f"SNR tensor must be 2D, got {snr_values_tensor.shape}"

    num_layers, num_heads = impact_scores_tensor.shape

    print(f"\nNumber of layers: {num_layers}")
    print(f"Number of heads: {num_heads}")
    

    # Flatten into list of (layer, head, impact, snr)
    head_stats = []
    for layer in range(num_layers):
        for head in range(num_heads):
            impact = impact_scores_tensor[layer][head].item()
            snr = snr_values_tensor[layer][head].item()
            head_stats.append((layer, head, impact, snr))

    print("\nAllosteric sensitivity analysis per attention head:")
    print("Layer | Head | Impact Score | SNR")
    print("-" * 40)
    for layer, head, impact, snr in head_stats:
        print(f"{layer:5d} | {head:4d} | {impact:11.3f} | {snr:6.2f}")

    # Calculate means
    mean_impact = np.mean([stat[2] for stat in head_stats])
    mean_snr = np.mean([stat[3] for stat in head_stats])

    # Select sensitive heads
    sensitive_heads = [
        (layer, head) for (layer, head, impact, snr) in head_stats
        if impact > mean_impact and snr > 2.0
    ]

    print(f"\nMost sensitive heads to allosteric sites {allosteric_res}:")
    print(f"(Impact > {mean_impact:.3f} and SNR > 2.0)")
    print(f"(Layer, Head) pairs: {sensitive_heads}")

    # Get attention maps
    attention_maps = analyzer.get_attention_maps(sequence)  # shape: [1, num_layers, num_heads, seq_len, seq_len]

    # Visualize averaged attention over sensitive heads for each layer
    heads_rep = [(0, 10), (0, 12)]

    for layer_idx, head_idx in heads_rep:
        analyzer.visualize_head_attention(
            attention_maps=attention_maps,
            allosteric_sites=allosteric_res,
            orthosteric_sites=orthosteric_res,
            pathway_sites=pathway_res,
            sequence=sequence,
            layer_idx=layer_idx,  # Use the layer index from sensitive_heads
            head_idx=head_idx  # Use the head index from sensitive_heads
        )

    # Save sensitive head info and attention
    output_dir = "attention_data"
    os.makedirs(output_dir, exist_ok=True)

    # Save metadata
    head_info = {
        'sensitive_heads': sensitive_heads,
        'impact_scores': {f"{l}_{h}": impact_scores_tensor[l][h].item() for (l, h) in sensitive_heads},
        'snr_values': {f"{l}_{h}": snr_values_tensor[l][h].item() for (l, h) in sensitive_heads}
    }
    torch.save(head_info, os.path.join(output_dir, 'sensitive_heads.pt'))

    # Save raw attention maps for sensitive heads
    sensitive_attention = []
    for (l, h) in sensitive_heads:
        att_map = attention_maps[0, l, h]
        sensitive_attention.append(att_map.unsqueeze(0))  # [1, seq_len, seq_len]

    if sensitive_attention:
        sensitive_attention_tensor = torch.cat(sensitive_attention, dim=0)  # [num_heads, seq_len, seq_len]
        torch.save(sensitive_attention_tensor, os.path.join(output_dir, 'sensitive_attention_maps.pt'))

    print(f"Saved sensitive head information and attention maps to {output_dir}/")

if __name__ == "__main__":
    main()
