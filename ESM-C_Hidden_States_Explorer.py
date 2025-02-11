"""
ESM-C Hidden States Explorer
===========================

This script explores the hidden states and embeddings from ESM-C (Evolutionary Scale Modeling - Continuous)
model to understand the internal representations that might be useful for allosteric site analysis.

The script examines:
- Hidden states shape and values
- Embeddings shape
- Layer-wise representations

This is part of a larger project to analyze allosteric sites in proteins using attention patterns,
based on the methodology from Dong et al. (2024).

Usage:
------
Run this script directly to test ESM-C with a sample protein sequence.
The script will print information about the model's internal representations.

Dependencies:
------------
- torch
- esm
- tokenizers==0.20.3

Authors:
--------
Aurelio A. Moya-Garcia
Date: February 11, 2025

References:
-----------
Dong et al. (2024). Allo-Allo: Data-efficient prediction of allosteric sites.
bioRxiv. DOI: pending
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
       
       output = self.model.logits(
           protein_tensor,
           LogitsConfig(
               sequence=True
               , return_embeddings=True
               , return_hidden_states=True
           )
       )
       
       # Examine hidden states shape and content
       if hasattr(output, 'hidden_states'):
           print("Hidden states shape:", output.hidden_states.shape)
       else:
           print("Hidden states not found in output.") 
          
           
       # Look at embeddings shape
       if hasattr(output, 'embeddings'):
           print("Embeddings shape:", output.embeddings.shape)
           
       # Examine first few values of hidden states
       if hasattr(output, 'hidden_states'):
           print("\nFirst few values of hidden states:")
           print(output.hidden_states[0, :5, :5])  # First layer, first 5 positions, first 5 dimensions
           
       return output

# Test the code
if __name__ == "__main__":
   # Print PyTorch device information
   print(f"PyTorch version: {torch.__version__}")
   print(f"Is MPS (M1) available? {torch.backends.mps.is_available()}")
   print(f"Is CUDA available? {torch.cuda.is_available()}")
   
   # Test with a sample sequence
   sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
   analyzer = AllosticHeadAnalyzer()
   output = analyzer.get_attention_maps(sequence)