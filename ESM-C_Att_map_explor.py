"""
ESM-C Attention Map Exploration
==============================

This script explores how to extract attention maps from ESM-C (Evolutionary Scale Modeling - Continuous) 
model, a state-of-the-art protein language model.

Purpose:
--------
- Test different ways to access attention information from ESM-C
- Print available attributes and outputs to understand the model's API
- Help determine the best way to analyze protein sequences for allosteric sites

Usage:
------
Run this script directly to test ESM-C with a sample protein sequence.
The script will print available attributes in the model's output.

Authors:
--------
Aurelio A. Moya-García
Date: February 11, 2025

References:
-----------
- ESM-C GitHub repository
- Dong et al. (2024) Allo-Allo paper
"""


import torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

import tokenizers
print(f"Tokenizers version: {tokenizers.__version__}")


class AllosticHeadAnalyzer:
    def __init__(self, threshold: float = 0.3, model_size: str = "600m"):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.model = ESMC.from_pretrained(f"esmc_{model_size}").to(self.device)
        self.threshold = threshold
        
    def get_attention_maps(self, sequence: str) -> torch.Tensor:
        protein = ESMProtein(sequence=sequence)
        protein_tensor = self.model.encode(protein)
        
        # Try different ways to get attention information
        output = self.model.logits(
            protein_tensor,
            LogitsConfig(
                sequence=True 
                , return_embeddings=True
                # Try additional parameters that might expose attention
                # return_representations=True, # does not work
                # return_contacts=True, does not work
                # , return_attention=True  # Let's try this one. Does not work
                , return_hidden_states=True
            )
        )
        
        # Let's see what's available in the output
        print("Available attributes in output:", dir(output))
        return output

        

# Test it
sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
analyzer = AllosticHeadAnalyzer()
output = analyzer.get_attention_maps(sequence)