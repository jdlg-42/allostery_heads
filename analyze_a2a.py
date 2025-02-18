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



from allosteric_analyzer import AllosticHeadAnalyzer

def main():
    # Adenosine A2A receptor 3VG9
    sequence = "MPIMGSSVYITVELAIAVLAILGNVLVCWAVWLNSNLQNVTNYFVVSLAAADIAVGVLAIPFAITISTGFCAACHGCLFIACFVLVLTQSSIFSLLAIAIDRYIAIRIPLRYNGLVTGTRAKGIIAICWVLSFAIGLTPMLGWNNCGQPKEGKQHSQGCGEGQVACLFEDVVPMNYMVYFNFFACVLVPLLLMLGVYLRIFLAARRQLKQMESQPLPGERARSTLQKEVHAAKSLAIIVGLFALCWLPLHIINCFTFFCPDCSHAPLWLMYLAIVLSHTNSVVNPFIYAYRIREFRQTFRKIIRSHVLRQQEPFKA"
    allosteric_sites = [85, 89, 246, 253]
    
    # Initialize analyzer
    analyzer = AllosticHeadAnalyzer(threshold=0.3)
    
    # Basic analysis
    results = analyzer.analyze_protein(sequence, allosteric_sites)
    
    # Get scores for each head
    impact_scores = results["impacts"].squeeze().tolist()
    
    print("\nAllosteric sensitivity scores per attention head:")
    for head_idx, score in enumerate(impact_scores):
        print(f"Head {head_idx}: {score:.3f}")
    
    # Identify most sensitive heads
    mean_score = sum(impact_scores) / len(impact_scores)
    sensitive_heads = [i for i, score in enumerate(impact_scores) if score > mean_score]
    
    print(f"\nMost sensitive heads to allosteric sites {allosteric_sites}:")
    print(f"Head indices: {sensitive_heads}")

if __name__ == "__main__":
    main()