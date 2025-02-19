"""
Allosteric Sensitivity Analysis in ESM2 Attention Heads
=====================================================

This module implements the methodology from Dong et al. (2024) for analyzing
allosteric sensitivity in protein language model attention heads, using ESM2.

Background
----------
Attention heads in Protein Language Models (PLMs) are specialized components that learn to focus on different aspects of relationships between amino acids in a protein sequence. Each attention head acts like a spotlight that can highlight specific connections or patterns between different positions in the sequence. For example, one head might focus on nearby amino acids that form local structural elements, while another might detect long-range interactions between distant parts of the protein. In technical terms, each attention head computes a weighted sum of all positions in the sequence for each position, where the weights (attention scores) indicate how much each position should influence the current position. These attention heads are organized in layers, with each layer containing multiple heads that work in parallel to capture different types of relationships. The combined output of these attention heads helps the model understand both local and global patterns in protein sequences, which is crucial for tasks like predicting protein structure, function, or in this case, allosteric sites.

Allostery is a mechanism where binding at one site affects protein function at
another site. Dong et al. 2024 shows that protein language models capture 
allosteric relationships in their attention heads. This code identifies which
attention heads are most sensitive to allosteric sites.

Classes
-------
AllosticHeadAnalyzer:
    Main class for analyzing allosteric sensitivity in ESM2 attention heads.

Example
-------
>>> from allosteric_analyzer import AllosticHeadAnalyzer
>>> 
>>> # Initialize analyzer
>>> analyzer = AllosticHeadAnalyzer(threshold=0.3)
>>> 
>>> # Example protein sequence (Adenosine A2A receptor)
>>> sequence = "MPIMGSSVYITVELAIAVLAILGNVLVCWAVWLNSNLQNVTNYFVVSLAAADIAVGVLAIPFAITISTGFCAACHGCLFIACFVLVLTQSSIFSLLAIAIDRYIAIRIPLRYNGLVTGTRAKGIIAICWVLSFAIGLTPMLGWNNCGQPKEGKQHSQGCGEGQVACLFEDVVPMNYMVYFNFFACVLVPLLLMLGVYLRIFLAARRQLKQMESQPLPGERARSTLQKEVHAAKSLAIIVGLFALCWLPLHIINCFTFFCPDCSHAPLWLMYLAIVLSHTNSVVNPFIYAYRIREFRQTFRKIIRSHVLRQQEPFKA"
>>> allosteric_sites = [85, 89, 246, 253]
>>> 
>>> # Analyze protein
>>> results = analyzer.analyze_protein(sequence, allosteric_sites)
>>> 
>>> # Get scores for each head
>>> impact_scores = results["impacts"].squeeze().tolist()
>>> 
>>> # Print results
>>> for head_idx, score in enumerate(impact_scores):
...     print(f"Head {head_idx}: {score:.3f}")
>>> 
>>> # Identify most sensitive heads
>>> mean_score = sum(impact_scores) / len(impact_scores)
>>> sensitive_heads = [i for i, score in enumerate(impact_scores) 
...                   if score > mean_score]
>>> print(f"Most sensitive heads: {sensitive_heads}")

    

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
    
    def seq_normal_compute_allosteric_impact(
        self,
        attention_maps: List[torch.Tensor],
        allosteric_sites: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute allosteric impact scores for each attention head. Sequence length normalization.
        
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

    def compute_allosteric_impact(self, attention_maps: List[torch.Tensor], allosteric_sites: List[int], n_random_trials: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute allosteric impact scores with proper random baseline, excluding allosteric sites from random selection
        
        Args:
            attention_maps: List of attention tensors from each layer
            allosteric_sites: List of allosteric site indices (1-based indexing)
            n_random_trials: Number of random samples to generate baseline
            
        Returns:
            Tuple of (allosteric_impacts, snr_values)
        """
        allo_sites = [site - 1 for site in allosteric_sites]
        n_allo_sites = len(allo_sites)
        impacts = []
        snrs = []
        
        for layer_attention in attention_maps:
            layer_impacts = []
            layer_snrs = []
            
            if layer_attention.dim() == 4:
                layer_attention = layer_attention.squeeze(0)
            
            for head in range(layer_attention.size(1)):
                attention = layer_attention[:, head]
                mask = attention > self.threshold
                seq_len = attention.size(0)
                
                # Calculate actual attention to allosteric sites
                w_allo = 0
                for site in allo_sites:
                    w_allo += torch.sum(attention[:, site][mask[:, site]]).item()
                
                # Create array of non-allosteric positions for random sampling
                non_allo_positions = np.array([i for i in range(seq_len) if i not in allo_sites])
                
                # Calculate random baseline with same number of sites
                random_w_values = []
                for _ in range(n_random_trials):
                    # Sample from non-allosteric positions only
                    random_sites = np.random.choice(non_allo_positions, size=n_allo_sites, replace=False)
                    random_w = 0
                    for site in random_sites:
                        random_w += torch.sum(attention[:, site][mask[:, site]]).item()
                    random_w_values.append(random_w)
                
                # Calculate mean and std of random baseline
                expected_random = np.mean(random_w_values)
                random_std = np.std(random_w_values)
                
                # Calculate impact and SNR
                impact = w_allo / expected_random if expected_random > 0 else 0
                snr = (w_allo - expected_random) / (random_std + 1e-10)
                
                layer_impacts.append(impact)
                layer_snrs.append(snr)
                
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

