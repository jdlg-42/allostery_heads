"""
Beta-2 Adrenergic Receptor (ADRB2) Allosteric Site Analysis
==========================================================

This script analyzes the allosteric sites of the human Beta-2 Adrenergic Receptor (ADRB2)
using the ESM2 protein language model attention patterns. It identifies which
attention heads are most sensitive to known allosteric sites.

The analysis focuses on four validated allosteric sites:
- Position 79: Key residue in receptor activation mechanism
- Position 282: Involved in conformational changes during signaling
- Position 318: Part of the intracellular binding domain 
- Position 319: Part of the intracellular binding domain

The Beta-2 Adrenergic Receptor (ADRB2) is a G protein-coupled receptor that mediates
the actions of catecholamines in multiple tissues. It plays a critical role in
cardiovascular and respiratory function by regulating heart rate, vascular tone,
and bronchodilation. The receptor's activity is modulated by both orthosteric
ligands (binding at the main binding pocket) and allosteric modulators that
bind at distinct sites to alter receptor conformation and function.

The script uses the AllosticHeadAnalyzer class to:
1. Process the protein sequence through ESM2
2. Analyze attention patterns around allosteric sites
3. Calculate sensitivity scores for each attention head
4. Identify heads that are most sensitive to allosteric communication

Usage
-----
$ python analyze_ADRB2.py

Output
------
The script prints:
- Sensitivity scores for each attention head
- List of heads with above-average sensitivity to allosteric sites
- Visualization of attention patterns

The script also saves:
- Sensitive head information to output files
- Attention maps for the most sensitive heads

Example Output
-------------
Allosteric sensitivity scores per attention head:
Head 0: 0.048
Head 1: 0.040
...
Head 19: 0.050

Most sensitive heads to allosteric sites [79, 282, 318, 319]:
Head indices: [0, 2, 3, 5, 6, 7, 9, 10, 12, 15, 18, 19]

References
----------
1. Wacker et al. (2010). Structural features for functional selectivity at
   serotonin receptors. Science, 330(6007), 1113-1116.
2. Dong et al. (2024). Allo-Allo: Data-efficient prediction of
   allosteric sites. bioRxiv. DOI: pending

Dependencies
-----------
- allosteric_analyzer.py
- torch
- fair-esm
- numpy
- matplotlib
- seaborn

Author
------
Created on: April 22, 2025
"""

import os
import torch
from allosteric_analyzer import AllosticHeadAnalyzer

def main():
    # Adenosine A2A receptor 3VG9
    sequence = "MGQPGNGSAFLLAPNGSHAPDHDVTQERDEVWVVGMGIVMSLIVLAIVFGNVLVITAIAKFERLQTVTNYFITSLACADLVMGLAVVPFGAAHILMKMWTFGNFWCEFWTSIDVLCVTASIETLCVIAVDRYFAITSPFKYQSLLTKNKARVIILMVWIVSGLTSFLPIQMHWYRATHQEAINCYANETCCDFFTNQAYAIASSIVSFYVPLVIMVFVYSRVFQEAKRQLQKIDKSEGRFHVQNLSQVEQDGRTGHGLRRSSKFCLKEHKALKTLGIIMGTFTLCWLPFFIVNIVHVIQDNLIRKEVYILLNWIGYVNSGFNPLIYCRSPDFRIAFQELLCLRRSSLKAYGNGYSSNGNTGEQSGYHVEQEKENKLLCEDLPGTEDFVGHQGTVPSDNIDSQGRNCSTNDSLL"
    allosteric_sites = [79, 282, 318, 319]
    
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

