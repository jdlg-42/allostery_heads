"""
ESM-C Attention Map Exploration
===============================

This script explores how to extract attention maps from the ESM-C (Evolutionary Scale Modeling - Continuous) 
model.

Purpose:
--------
- Test different ways to access attention information from ESM-C
- Print available attributes and outputs to understand the model's API
- Help determine the best way to analyze protein sequences for allosteric sites

Usage:
------
Run this script directly to test ESM-C with a sample protein sequence.
The script will print available attributes in the model's output and the shape of hidden states.

Authors:
--------
Aurelio A. Moya-Garcia
Date: February 11, 2025

References:
-----------
- ESM-C GitHub repository
- Dong et al. (2024) Allo-Allo paper
"""


import torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

class AllosticHeadAnalyzer:
    def __init__(self, threshold: float = 0.3, model_size: str = "600m"):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.model = ESMC.from_pretrained(f"esmc_{model_size}").to(self.device)
        self.threshold = threshold
        
    def get_attention_maps(self, sequence: str):
        protein = ESMProtein(sequence=sequence)
        protein_tensor = self.model.encode(protein)
        
        # Get logits output with embeddings and hidden states
        output = self.model.logits(
            protein_tensor,
            LogitsConfig(
                sequence=True,
                return_embeddings=True,
                return_hidden_states=True
            )
        )
        
        # Print available attributes in the output
        print("Available attributes in output:", dir(output))
        
        # Print the shape of hidden states to understand their structure
        if hasattr(output, 'hidden_states'):
            hidden_states = output.hidden_states
            print("Hidden states shape:", hidden_states.shape)
        
        return output

# Test it
sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
analyzer = AllosticHeadAnalyzer()
output = analyzer.get_attention_maps(sequence)