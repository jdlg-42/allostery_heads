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
    
    # This function is replaced with the new version below
    def old_compute_allosteric_impact(
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
                # print(f"Attention shape: {attention.shape}")  # Debug print
                
                # Calculate w_allo (eq. 1 from paper)
                w_allo = 0
                mask = attention > self.threshold
                for site in allo_sites:
                    w_allo += torch.sum(attention[:, site][mask[:, site]]).item()    
                
                # Calculate total activity w (eq. 2 from paper)
                w_total = torch.sum(attention > self.threshold).item()
                
                # Calculate impact score p (eq. after eq. 2 in paper)
                # Normalize impact by sequence length factor
                seq_len = attention_maps[0].size(-1)
                base_factor = seq_len / len(allosteric_sites)  # Ratio of sequence length to number of sites
                length_factor = np.log2(base_factor)  # Logarithmic scaling
    
                impact = (w_allo / w_total) * length_factor if w_total > 0 else 0
                layer_impacts.append(impact)
                layer_snrs.append(impact)
            
            impacts.append(layer_impacts)
            snrs.append(layer_snrs)
            
        impacts = torch.tensor(impacts)
        snrs = torch.tensor(snrs)
        
        return impacts, snrs
    
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
                
                # Calculate w_allo (eq. 1 from paper)
                w_allo = 0
                mask = attention > self.threshold
                for site in allo_sites:
                    w_allo += torch.sum(attention[:, site][mask[:, site]]).item()    
                
                # Calculate total activity w (eq. 2 from paper)
                w_total = torch.sum(attention > self.threshold).item()
                
                # Calculate impact score p (eq. after eq. 2 in paper)
                # Normalize impact by sequence length factor
                seq_len = attention_maps[0].size(-1)
                base_factor = seq_len / len(allosteric_sites)
                length_factor = np.log2(base_factor)
                
                impact = (w_allo / w_total) * length_factor if w_total > 0 else 0
                layer_impacts.append(impact)
                layer_snrs.append(impact)
            
            impacts.append(layer_impacts)
            snrs.append(layer_snrs)
        
        return torch.tensor(impacts), torch.tensor(snrs)

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

        
    def analyze_sequence_length_effect(self, sequence: str, allosteric_sites: List[int]) -> Dict:
        """Analyze how sequence length affects attention scores"""
        seq_len = len(sequence)
        num_sites = len(allosteric_sites)
        
        # Basic statistics
        stats = {
            "sequence_length": seq_len,
            "num_allosteric_sites": num_sites,
            "sites_ratio": num_sites / seq_len,
            "expected_random_attention": 1.0 / seq_len
        }
        
        # Get attention maps
        attention_maps = self.get_attention_maps(sequence)
        
        # Convert to 0-based indexing
        allo_sites = [site - 1 for site in allosteric_sites]
        
        # Take first layer and squeeze batch dimension if present
        layer_attention = attention_maps[0]  # Shape: [num_heads, seq_len, seq_len]
        if layer_attention.dim() == 4:
            layer_attention = layer_attention.squeeze(0)
        
        # Create masks for allosteric sites
        allo_mask = torch.zeros(layer_attention.size(-1), dtype=torch.bool, device=layer_attention.device)
        allo_mask[allo_sites] = True
        non_allo_mask = ~allo_mask
        
        # Calculate attention statistics
        # 1. First get attention scores above threshold
        threshold_mask = layer_attention > self.threshold
        
        # 2. Calculate mean attention for allosteric sites (averaging across heads)
        allo_attention = 0
        non_allo_attention = 0
        
        for head in range(layer_attention.size(0)):
            head_attention = layer_attention[head]
            head_threshold_mask = threshold_mask[head]
            
            # For allosteric sites
            if allo_mask.any():
                valid_attention = head_attention[head_threshold_mask & allo_mask.unsqueeze(0)]
                if len(valid_attention) > 0:
                    allo_attention += valid_attention.mean().item()
                    
            # For non-allosteric sites
            if non_allo_mask.any():
                valid_attention = head_attention[head_threshold_mask & non_allo_mask.unsqueeze(0)]
                if len(valid_attention) > 0:
                    non_allo_attention += valid_attention.mean().item()
        
        # Average across heads
        num_heads = layer_attention.size(0)
        allo_attention /= num_heads
        non_allo_attention /= num_heads
        
        stats.update({
            "mean_attention_allosteric": allo_attention,
            "mean_attention_non_allosteric": non_allo_attention,
            "attention_ratio": allo_attention / non_allo_attention if non_allo_attention > 0 else 0
        })
        
        return stats    
    
    def analyze_distance_effect(self, attention_maps, allosteric_sites, head_idx):
        """Analyze how attention varies with distance from allosteric sites"""
        # Get attention for specific head and reshape if needed
        head_attention = attention_maps[0, head_idx]  # Shape: [seq_len, seq_len]
        if head_attention.dim() > 2:
            head_attention = head_attention.squeeze()  # Remove extra dimensions
        
        distances = {}
        
        for site in allosteric_sites:
            # Get attention from all positions to this allosteric site
            site_attention = head_attention[:, site-1]  # 0-based indexing
            
            # Calculate distances and attention values
            for pos in range(len(site_attention)):
                distance = abs(pos - (site-1))
                # Get mean attention value across all positions
                attention_value = site_attention[pos].mean().cpu().item()
                
                if distance not in distances:
                    distances[distance] = []
                distances[distance].append(attention_value)
        
        # Average attention by distance
        avg_attention = {d: np.mean(v) for d, v in distances.items()}
        return avg_attention


    def calculate_allosteric_attention_scores(self, sequence: str, allosteric_sites: List[int], sensitive_heads: List[int]) -> Dict[int, float]:
        """
        Calculate attention scores for each amino acid based on how much they attend to allosteric sites
        in allostery-sensitive heads.
        
        Args:
            sequence: Protein sequence
            allosteric_sites: List of allosteric site positions (1-based indexing)
            sensitive_heads: List of head indices identified as allosteric-sensitive
            
        Returns:
            Dictionary mapping position (1-based) to attention score
        """
        attention_maps = self.get_attention_maps(sequence)
        position_scores = {}
        
        # Convert to 0-based indexing
        allo_sites = [site - 1 for site in allosteric_sites]
        
        # For each position in the sequence
        for pos in range(len(sequence)):
            total_attention = 0
            
            # Average attention across sensitive heads
            for head in sensitive_heads:
                head_attention = attention_maps[0, head]
                
                # Get attention from this position to all allosteric sites
                for site in allo_sites:
                    total_attention += head_attention[pos, site].item()
            
            # Average across heads and sites
            avg_attention = total_attention / (len(sensitive_heads) * len(allo_sites))
            position_scores[pos + 1] = avg_attention  # Convert back to 1-based indexing
        
        return position_scores


    def analyze_window_scores(self, attention_maps: torch.Tensor, allosteric_sites: List[int], window_size: int = 5) -> Dict[int, float]:
        """
        Analyze attention patterns in windows around allosteric sites.
        Returns head scores based on attention to allosteric regions.
        """
        head_scores = {}
        
        print(f"\nDebug - Attention maps shape: {attention_maps.shape}")
        
        # Reshape attention maps if needed
        if attention_maps.dim() == 5:  # [1, 33, 20, 318, 318]
            attention_maps = attention_maps.squeeze(0)  # Now [33, 20, 318, 318]
        
        for head in range(self.num_heads):
            # Get the correct attention matrix for this head
            head_attention = attention_maps[0, head]  # Get first layer, specific head
            print(f"\nDebug - Head {head} attention shape: {head_attention.shape}")
            print(f"Debug - Head {head} attention range: [{head_attention.min().item():.6f}, {head_attention.max().item():.6f}]")
            
            total_score = 0
            valid_sites = 0
            
            for site in allosteric_sites:
                # Define window boundaries (convert to 0-based indexing)
                site_idx = site - 1
                start = max(0, site_idx - window_size)
                end = min(head_attention.shape[0], site_idx + window_size + 1)
                
                if end > start:
                    # Get window attention and apply threshold
                    window_attention = head_attention[start:end, start:end]
                    mask = window_attention > self.threshold
                    filtered_attention = window_attention[mask]
                    
                    print(f"Debug - Site {site}:")
                    print(f"Window shape: {window_attention.shape}")
                    print(f"Window boundaries: [{start}, {end}]")
                    print(f"Values above threshold: {filtered_attention.numel()}")
                    
                    if filtered_attention.numel() > 0:
                        window_score = filtered_attention.mean().item()
                        print(f"Window score: {window_score:.6f}")
                        total_score += window_score
                        valid_sites += 1
            
            # Average across valid allosteric sites
            if valid_sites > 0:
                head_scores[head] = total_score / valid_sites
            else:
                head_scores[head] = 0.0
            
            print(f"Debug - Head {head} final score: {head_scores[head]:.6f}")
        
        return head_scores



# Example usage:
if __name__ == "__main__":
    # Example sequence and allosteric sites
    # sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    # allosteric_sites = [10, 15, 20]  # 1-based indices of allosteric sites

    # Adenosine A2A receptor 3VG9
    sequence = "MPIMGSSVYITVELAIAVLAILGNVLVCWAVWLNSNLQNVTNYFVVSLAAADIAVGVLAIPFAITISTGFCAACHGCLFIACFVLVLTQSSIFSLLAIAIDRYIAIRIPLRYNGLVTGTRAKGIIAICWVLSFAIGLTPMLGWNNCGQPKEGKQHSQGCGEGQVACLFEDVVPMNYMVYFNFFACVLVPLLLMLGVYLRIFLAARRQLKQMESQPLPGERARSTLQKEVHAAKSLAIIVGLFALCWLPLHIINCFTFFCPDCSHAPLWLMYLAIVLSHTNSVVNPFIYAYRIREFRQTFRKIIRSHVLRQQEPFKA"
    allosteric_sites = [85, 89, 246, 253]
    
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



    #  # Analyze sequence length effect
    # stats = analyzer.analyze_sequence_length_effect(sequence, allosteric_sites)
    
    # print("\nSequence Length Analysis:")
    # print(f"Sequence length: {stats['sequence_length']}")
    # print(f"Number of allosteric sites: {stats['num_allosteric_sites']}")
    # print(f"Ratio of sites to length: {stats['sites_ratio']:.3f}")
    # print(f"Expected random attention: {stats['expected_random_attention']:.3f}")
    # print(f"Mean attention to allosteric sites: {stats['mean_attention_allosteric']:.3f}")
    # print(f"Mean attention to non-allosteric sites: {stats['mean_attention_non_allosteric']:.3f}")
    # print(f"Attention ratio (allosteric/non-allosteric): {stats['attention_ratio']:.3f}")


    
    # # Analyze distance effect for most sensitive heads
    # attention_maps = analyzer.get_attention_maps(sequence)  # Get attention maps if not already available
    # # For a sensitive head (e.g., head 2 which had high score)
    # avg_attention = analyzer.analyze_distance_effect(attention_maps, allosteric_sites, 2)
    
    #  # In main section:
    # print("\nDistance Effect Analysis for Head 2:")
    # print("Average attention by distance from allosteric sites:")
    # distances_to_show = sorted(avg_attention.items())[:20]  # Show more distances
    # print("\nDistance | Attention | Visualization")
    # print("-" * 50)
    # for distance, attention in distances_to_show:
    #     bar = "█" * int(attention * 100)  # Visualize with bars
    #     print(f"{distance:8d} | {attention:.4f}  | {bar}")






    # The code is calculating how much each head "pays attention" to your specified allosteric sites (positions 10, 15, and 20)
    # Higher scores (closer to 1.0): The head pays more attention to the allosteric sites
    # Lower scores (closer to 0.0): The head pays less attention to the allosteric sites
    # I guess I can get the actual attention values for each head that has a high score. Each attention value should correspond to an amino acid in the sequence, so I can see which amino acids are most important for the head's decision.
    
    # We can look at the actual attentio paterns for the most sensitive heads
    # 1. Look at the actual attention patterns for the most sensitive heads


        # attention_maps = analyzer.get_attention_maps(sequence)
        # for head in sensitive_heads:
        #     head_attention = attention_maps[0, head]
        #     print(f"\nHead {head} attention analysis:")
            
        #     # For each allosteric site
        #     for site, pos in zip([10, 15, 20], [9, 14, 19]):  # Convert to 0-based indexing
        #         # Get attention scores for this site
        #         site_attention = head_attention[:, pos]
                
        #         # Find top 5 positions attending to this site
        #         top_values, top_indices = torch.topk(site_attention, 5)
                
        #         print(f"\nSite {site} analysis:")
        #         print(f"Max attention: {site_attention.max().item():.3f}")
        #         print("Top 5 attending positions:")
                
        #         # Convert tensors to numpy arrays and iterate
        #         indices = top_indices.cpu().numpy()
        #         values = top_values.cpu().numpy()
                
        #         for idx, value in zip(indices.flatten(), values.flatten()):
        #             residue = sequence[idx] if idx < len(sequence) else "N/A"
        #             print(f"Position {idx+1}: {value.item():.3f} (residue: {residue})")













    # # I want to trace the allosteric communication pathways by looking at which amino acids are paying attention to the allosteric sites. The calculate_allosteric_attention_scores method  calculates this.

    # This will:

    # Calculate an "allostery score" for each position based on how much it attends to the allosteric sites
    # Use only the heads that were identified as allosteric-sensitive
    # Show the top positions that might be involved in allosteric communication
    # You can then use these scores to:

    # Map them onto the protein structure
    # Identify potential communication pathways
    # Compare with known orthosteric sites
    # Look for patterns of connectivity between allosteric and orthosteric sites

    # # Calculate allosteric attention scores
    # allo_scores = analyzer.calculate_allosteric_attention_scores(sequence, allosteric_sites, sensitive_heads)

    # # Print results sorted by attention score
    # print("\nAllosteric Communication Analysis:")
    # print("\nPosition | AA | Attention Score | Visualization")
    # print("-" * 60)

    # # Sort positions by attention score
    # sorted_positions = sorted(allo_scores.items(), key=lambda x: x[1], reverse=True)

    # # Print top 20 positions
    # for pos, score in sorted_positions[:20]:
    #     aa = sequence[pos-1]
    #     bar = "█" * int(score * 100)  # Visual representation of score
    #     print(f"{pos:8d} | {aa:2s} | {score:.4f} | {bar}")

    # # You might want to save these scores for visualization in PyMOL or other tools
    # scores_for_visualization = {pos: score for pos, score in sorted_positions}



    # # Get attention maps for window analysis
    # attention_maps = analyzer.get_attention_maps(sequence)
    
    # # Analyze windows around allosteric sites
    # window_scores = analyzer.analyze_window_scores(attention_maps, allosteric_sites, window_size=5)
    
    # # Print window analysis results
    # print("\nWindow Analysis Results:")
    # print("Head | Window Score | Visualization")
    # print("-" * 50)
    
    # # Sort heads by window score
    # sorted_heads = sorted(window_scores.items(), key=lambda x: x[1], reverse=True)
    # for head, score in sorted_heads:
    #     # Handle potential NaN or very small values
    #     bar_length = max(0, int(score * 1000)) if not np.isnan(score) else 0
    #     bar = "█" * bar_length
    #     print(f"{head:4d} | {score:9.6f} | {bar}")
    
    # # Identify sensitive heads (above mean)
    # mean_window_score = sum(window_scores.values()) / len(window_scores)
    # sensitive_heads_window = [head for head, score in window_scores.items() 
    #                         if score > mean_window_score]
    
    # print(f"\nMost sensitive heads (window analysis):")
    # print(f"Head indices: {sensitive_heads_window}")
    
    # # Compare with previous method
    # print("\nComparison of methods:")
    # print(f"Original method sensitive heads: {sensitive_heads}")
    # print(f"Window method sensitive heads: {sensitive_heads_window}")
    # print(f"Heads identified by both methods: {set(sensitive_heads) & set(sensitive_heads_window)}")


   
   
   
    # # Analyze multiple proteins
    # sequences = [sequence] * 3  # Example with multiple copies
    # allosteric_sites = [allosteric_sites] * 3
    
    # df = analyzer.analyze_dataset(sequences, allosteric_sites)
    # print("\nTop 5 attention heads by SNR:")
    # print(df.nlargest(5, "snr"))