

"""
ESM-C Hidden States Explorer and Allosteric Analysis
=================================================

This script explores ESM-C hidden states and performs allosteric site analysis.
"""

import torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
import numpy as np
from typing import List, Dict
import matplotlib.pyplot as plt

class ESMCAnalyzer:
    def __init__(self, threshold: float = 0.3, model_size: str = "600m"):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.model = ESMC.from_pretrained(f"esmc_{model_size}").to(self.device)
        
    def analyze_sequence(self, sequence: str) -> Dict:
        """Analyze a protein sequence"""
        # Get model outputs
        protein = ESMProtein(sequence=sequence)
        protein_tensor = self.model.encode(protein)
        output = self.model.logits(
            protein_tensor,
            LogitsConfig(
                sequence=True,
                return_embeddings=True,
                return_hidden_states=True
            )
        )
        
        # Print basic information
        print("\nModel Output Information:")
        print(f"Hidden states shape: {output.hidden_states.shape}")
        print(f"Embeddings shape: {output.embeddings.shape}")
        
        # Analyze position dynamics
        hidden_states = output.hidden_states
        
        # Calculate changes between layers for each position
        layer_changes = []
        for layer in range(1, len(hidden_states)):
            change = torch.norm(hidden_states[layer] - hidden_states[layer-1], dim=-1)
            layer_changes.append(change)
        
        layer_changes = torch.stack(layer_changes)
        position_dynamics = torch.mean(layer_changes, dim=0).squeeze()
        
        # Calculate pairwise interactions from final layer
        final_layer = hidden_states[-1].squeeze()
        norm = torch.norm(final_layer, dim=1, keepdim=True)
        normalized = final_layer / norm
        interactions = torch.mm(normalized, normalized.t())
        
        # Identify potential allosteric sites
        # Positions with high dynamics and strong interactions
        dynamics_score = position_dynamics
        interaction_score = torch.mean(interactions, dim=1)
        combined_score = dynamics_score * interaction_score
        
        # Select top 10% as potential allosteric sites
        threshold = torch.quantile(combined_score, 0.9)
        potential_sites = torch.where(combined_score > threshold)[0].tolist()
        
        print("\nAnalysis Results:")
        print(f"Potential allosteric sites (0-based indices): {potential_sites}")
        
        # Visualize the results
        plt.figure(figsize=(12, 4))
        plt.plot(combined_score.cpu().numpy())
        plt.title("Position Scores (Higher Values Suggest Potential Allosteric Sites)")
        plt.xlabel("Sequence Position")
        plt.ylabel("Score")
        plt.show()
        
        return {
            'hidden_states': hidden_states,
            'position_dynamics': position_dynamics,
            'interactions': interactions,
            'potential_sites': potential_sites
        }

# Test the code
if __name__ == "__main__":
    # Print PyTorch device information
    print(f"PyTorch version: {torch.__version__}")
    print(f"Is MPS (M1) available? {torch.backends.mps.is_available()}")
    print(f"Is CUDA available? {torch.cuda.is_available()}")
    
    # Test sequence
    sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    
    # Initialize and run analyzer
    analyzer = ESMCAnalyzer()
    results = analyzer.analyze_sequence(sequence)