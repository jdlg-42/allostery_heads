"""
Allosteric Sensitivity Analysis in ESM2 Attention Heads
=====================================================

This module implements the methodology from Dong et al. (2024) for analyzing
allosteric sensitivity in protein language model attention heads, using ESM2.

Background
----------
Attention heads in Protein Language Models (PLMs) are specialized components that learn to focus on different aspects of relationships between amino acids in a protein sequence. Each attention head acts like a spotlight that can highlight specific connections or patterns between different positions in the sequence. For example, one head might focus on nearby amino acids that form local structural elements, while another might detect long-range interactions between distant parts of the protein. In technical terms, each attention head computes a weighted sum of all positions in the sequence for each position, where the weights (attention scores) indicate how much each position should influence the current position. These attention heads are organized in layers, with each layer containing multiple heads that work in parallel to capture different types of relationships. The combined output of these attention heads helps the model understand both local and global patterns in protein sequences, which is crucial for tasks like predicting protein structure, function, or in this case, allosteric sites.

Allostery is a mechanism where binding at one site affects protein function at
another site. The paper shows that protein language models (PLMs) capture 
allosteric relationships in their attention heads. This code identifies which
attention heads are most sensitive to allosteric sites.

Key Features
-----------
- Extracts attention maps from ESM2 for protein sequences
- Calculates allosteric impact scores using equations from Dong et al.
- Identifies attention heads most sensitive to allosteric relationships
- Supports both single protein and batch analysis

Dependencies
-----------
- torch
- fair-esm
- numpy
- pandas
- tqdm

Authors:
--------
Aurelio A. Moya-García
Date: February 10, 2025

References
----------
Dong et al. (2024). Allo-Allo: Data-efficient prediction of allosteric sites.
bioRxiv. DOI: pending
"""

import torch
from esm import pretrained
import numpy as np
from typing import List, Dict, Tuple
import pandas as pd
from tqdm import tqdm

class AllosticHeadAnalyzer:
    def __init__(self, threshold: float = 0.3, model_name: str = "esm2_t33_650M_UR50D"):
        """
        Initialize the analyzer with ESM2 model
        
        Args:
            threshold: Minimum score for attention values to be considered significant
            model_name: Name of the ESM2 model to use
        """
        # Set up device for M1 Mac (MPS) or fallback to CPU
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Initialize ESM2 model
        self.model, self.alphabet = pretrained.load_model_and_alphabet(model_name)
        self.model = self.model.to(self.device)
        self.threshold = threshold
        
        # Get model dimensions - these will be initialized after first run
        self.num_layers = self.model.num_layers
        self.num_heads = self.model.attention_heads
        
    def get_attention_maps(self, sequence: str) -> torch.Tensor:
        """
        Get attention maps for a protein sequence
        
        Args:
            sequence: Amino acid sequence
            
        Returns:
            Tensor of shape [num_layers, num_heads, seq_len, seq_len]
        """
        # Tokenize the sequence
        batch_converter = self.alphabet.get_batch_converter()
        batch_labels, batch_strs, batch_tokens = batch_converter([("protein", sequence)])
        batch_tokens = batch_tokens.to(self.device)
        
        # Get logits output with attention maps
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[self.num_layers], return_contacts=True)
        
        attentions = results["attentions"]
        
        return attentions
    
    def compute_allosteric_impact(
        self,
        attention_maps: List[torch.Tensor],
        allosteric_sites: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute allosteric impact scores for each attention head
        
        Args:
            attention_maps: List of attention tensors from each layer
            allosteric_sites: List of allosteric site indices (1-based indexing)
            
        Returns:
            Tuple of (allosteric_impacts, snr_values)
        """
        # Convert to 0-based indexing
        allo_sites = [site - 1 for site in allosteric_sites]
        
        impacts = []
        snrs = []
        
        # Compute impact for each layer and head
        for layer_attention in attention_maps:
            layer_impacts = []
            layer_snrs = []
            
            # Squeeze out batch dimension if present
            if layer_attention.dim() == 4:
                layer_attention = layer_attention.squeeze(0)
            
            for head in range(layer_attention.size(1)):
                attention = layer_attention[:, head]
                print(f"Attention shape: {attention.shape}")  # Debug print
                
                # Calculate w_allo (eq. 1 from paper)
                w_allo = 0
                mask = attention > self.threshold
                for site in allo_sites:
                    w_allo += torch.sum(attention[:, site][mask[:, site]]).item()    
                
                # Calculate total activity w (eq. 2 from paper)
                w_total = torch.sum(attention > self.threshold).item()
                
                # Calculate impact score p (eq. after eq. 2 in paper)
                impact = w_allo / w_total if w_total > 0 else 0
                layer_impacts.append(impact)
                layer_snrs.append(impact)
            
            impacts.append(layer_impacts)
            snrs.append(layer_snrs)
            
        impacts = torch.tensor(impacts)
        snrs = torch.tensor(snrs)
        
        return impacts, snrs
    
    def analyze_protein(
        self,
        sequence: str,
        allosteric_sites: List[int]
    ) -> Dict[str, torch.Tensor]:
        """
        Analyze a single protein sequence
        
        Args:
            sequence: Amino acid sequence
            allosteric_sites: List of allosteric site indices (1-based)
            
        Returns:
            Dictionary with impact scores and SNR values
        """
        attention_maps = self.get_attention_maps(sequence)
        impacts, snrs = self.compute_allosteric_impact(attention_maps, allosteric_sites)
        
        return {
            "impacts": impacts,
            "snrs": snrs
        }
    
    def analyze_dataset(
        self,
        sequences: List[str],
        allosteric_sites: List[List[int]]
    ) -> pd.DataFrame:
        """
        Analyze a dataset of proteins
        
        Args:
            sequences: List of protein sequences
            allosteric_sites: List of lists containing allosteric sites for each sequence
            
        Returns:
            DataFrame with impact scores for each layer-head combination
        """
        all_impacts = []
        
        for seq, sites in tqdm(zip(sequences, allosteric_sites), total=len(sequences)):
            results = self.analyze_protein(seq, sites)
            all_impacts.append(results["impacts"])
            
        # Stack impacts from all proteins
        impacts_tensor = torch.stack(all_impacts)
        
        # Calculate mean and std across proteins
        mean_impacts = torch.mean(impacts_tensor, dim=0)
        std_impacts = torch.std(impacts_tensor, dim=0)
        snr = mean_impacts / (std_impacts + 1e-10)
        
        # Create DataFrame
        rows = []
        for layer in range(self.num_layers):
            for head in range(self.num_heads):
                rows.append({
                    "layer": layer,
                    "head": head,
                    "mean_impact": mean_impacts[layer, head].item(),
                    "std_impact": std_impacts[layer, head].item(),
                    "snr": snr[layer, head].item()
                })
                
        return pd.DataFrame(rows)

# Example usage:
if __name__ == "__main__":
    # Example sequence and allosteric sites
    sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    allosteric_sites = [10, 15, 20]  # 1-based indices of allosteric sites
    
    # Initialize analyzer
    analyzer = AllosticHeadAnalyzer(threshold=0.3)
    
    # # Analyze single protein
    # results = analyzer.analyze_protein(sequence, allosteric_sites)
    # print("Impact scores shape:", results["impacts"].shape)
    # print("SNR values shape:", results["snrs"].shape)
    
   # Analyze single protein
    results = analyzer.analyze_protein(sequence, allosteric_sites)
    
    # Get scores for each head
    impact_scores = results["impacts"].squeeze().tolist()
    
    print("\nAllosteric sensitivity scores per attention head:")
    for head_idx, score in enumerate(impact_scores):
        print(f"Head {head_idx}: {score:.3f}")
    
    # Identify most sensitive heads (above mean)
    mean_score = sum(impact_scores) / len(impact_scores)
    sensitive_heads = [i for i, score in enumerate(impact_scores) if score > mean_score]
    
    print(f"\nMost sensitive heads to allosteric sites {allosteric_sites}:")
    print(f"Head indices: {sensitive_heads}")
   
# The code is calculating how much each head "pays attention" to your specified allosteric sites (positions 10, 15, and 20)
# Higher scores (closer to 1.0): The head pays more attention to the allosteric sites
# Lower scores (closer to 0.0): The head pays less attention to the allosteric sites
# I guess I can get the actual attention values for each head that has a high score. Each attention value should correspond to an amino acid in the sequence, so I can see which amino acids are most important for the head's decision.
   
   
   
   
   
   
    # # Analyze multiple proteins
    # sequences = [sequence] * 3  # Example with multiple copies
    # allosteric_sites = [allosteric_sites] * 3
    
    # df = analyzer.analyze_dataset(sequences, allosteric_sites)
    # print("\nTop 5 attention heads by SNR:")
    # print(df.nlargest(5, "snr"))