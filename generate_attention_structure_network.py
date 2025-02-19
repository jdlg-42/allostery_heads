import torch
import networkx as nx
import matplotlib.pyplot as plt
from ProteinStructureNetwork import ProteinStructureNetwork
import os
import argparse
from typing import Dict, List
import seaborn as sns
import numpy as np

"""
Protein Structure Network Analysis with ESM2 Attention
===================================================

This script combines protein structure information from PDB files with attention
patterns from ESM2 attention heads identified as sensitive to allosteric sites.
It builds a protein structure network where edges represent residues within a
distance threshold, and annotates these edges with attention scores.

Requirements
-----------
- Saved attention maps from analyze_a2a.py in output directory:
  - sensitive_heads.pt: Information about identified sensitive heads
  - sensitive_attention_maps.pt: Attention patterns from these heads

Dependencies
-----------
- torch
- networkx
- matplotlib
- BioPython
- numpy
- seaborn

Usage
-----
First, run analyze_a2a.py to generate attention data:
    $ python analyze_a2a.py

Then run this script:
    $ python generate_attention_structure_network.py --pdb 3vg9.pdb --chain A

Arguments
---------
--pdb : str
    Path to PDB file (required)
--chain : str
    Chain ID to analyze (default: 'A')
--distance : float
    Distance threshold in Angstroms (default: 5.0)
--output : str
    Directory containing attention data (default: 'attention_data')

Output
------
For each sensitive head:
- Network visualization saved as PNG
- Basic network statistics
- Attention score distribution

Example
-------
# Basic usage
$ python generate_attention_structure_network.py --pdb 3vg9.pdb

# Specify chain and distance threshold
$ python generate_attention_structure_network.py --pdb 3vg9.pdb --chain B --distance 6.0

# Custom output directory
$ python generate_attention_structure_network.py --pdb 3vg9.pdb --output results

Author
------
Aurelio A. Moya-García
Date: February 19, 2025
"""


def load_attention_data(output_dir: str) -> Dict:
    """Load saved attention maps and head information"""
    head_info = torch.load(os.path.join(output_dir, 'sensitive_heads.pt'))
    attention_maps = torch.load(os.path.join(output_dir, 'sensitive_attention_maps.pt'))
    return {'head_info': head_info, 'attention_maps': attention_maps}

def plot_network_with_attention(G: nx.Graph, head_idx: int, save_path: str = None):
    """Plot network with attention scores as edge colors"""
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Get attention scores
    edge_attention = [G[u][v]['attention'] for u,v in G.edges()]
    
    plt.figure(figsize=(12, 12))
    nx.draw_networkx_nodes(G, pos, node_size=100)
    edges = nx.draw_networkx_edges(G, pos, edge_color=edge_attention, 
                                 edge_cmap=plt.cm.viridis,
                                 width=2, alpha=0.7)
    plt.colorbar(edges)
    plt.title(f'Protein Structure Network - Head {head_idx}\nEdge colors show attention scores')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Analyze protein structure network with attention')
    parser.add_argument('--pdb', required=True, help='Path to PDB file')
    parser.add_argument('--chain', default='A', help='Chain ID to analyze')
    parser.add_argument('--distance', type=float, default=5.0, help='Distance threshold (Å)')
    parser.add_argument('--output', default='attention_data', help='Output directory')
    args = parser.parse_args()

    # Load attention data
    data = load_attention_data(args.output)
    head_info = data['head_info']
    attention_maps = data['attention_maps']
    
    # Initialize network builder
    psn = ProteinStructureNetwork(distance_threshold=args.distance)
    
    # Build network from PDB
    print(f"Building protein structure network from {args.pdb}...")
    network = psn.build_network(args.pdb, chain_id=args.chain)
    print(f"Network built: {network.number_of_nodes()} nodes, {network.number_of_edges()} edges")
    
    # Analyze each sensitive head
    for idx, head_idx in enumerate(head_info['head_indices']):
        print(f"\nAnalyzing head {head_idx}")
        print(f"Impact score: {head_info['impact_scores'][idx]:.3f}")
        print(f"SNR: {head_info['snr_values'][idx]:.2f}")
        
        # Get network with attention scores
        attention_network = psn.get_attention_annotated_network(idx, attention_maps)
        
        # Plot and save network
        plot_network_with_attention(attention_network, head_idx, 
                                  save_path=os.path.join(args.output, f'network_head_{head_idx}.png'))
        
        # Basic network analysis
        print("\nNetwork statistics:")
        print(f"Average attention score: {np.mean([d['attention'] for _,_,d in attention_network.edges(data=True)]):.3f}")
        print(f"Max attention score: {np.max([d['attention'] for _,_,d in attention_network.edges(data=True)]):.3f}")

if __name__ == "__main__":
    main()