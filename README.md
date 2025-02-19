# ESM Allostery Analysis

This repository contains tools for analyzing allosteric sensitivity in ESM2 protein language model attention heads. It implements and extends the methodology from Dong et al. (2024) to identify which attention heads are most sensitive to known allosteric sites.

## Allosteric Sensitivity
Attention maps in protein language models (PLMs) are matrices that represent the attention scores between different positions in a protein sequence. These scores indicate how much each position in the sequence should influence other positions, allowing the model to focus on specific relationships and patterns within the sequence.

In more detail:

- Attention Mechanism: The attention mechanism in PLMs computes a weighted sum of all positions in the sequence for each position. The attention mechanism determines the weights (attention scores) and indicate the importance of each position relative to others.

- Attention Heads: PLMs typically have multiple attention heads in each layer. Each attention head learns to focus on different aspects of the sequence, such as local structural elements or long-range interactions.

- Attention Maps: The attention scores for each head are organized into attention maps, which are tensors of shape [num_layers, num_heads, seq_len, seq_len]. Each element in the attention map represents the attention score between two positions in the sequence.

Attention maps can be used to interpret how the model understands the relationships between different amino acids in the protein sequence. For example, high attention scores between distant positions indicate long-range interactions important for the protein's function.

Each attention map represents how much each position in the sequence attends to every other position. For each layer and head (ESM2 has 33 layers and 20 heads in each layer), you get a matrix where entry (i,j) shows how much position i pays attention to position j.

For each head, we have an attention matrix where:
- i represents the source position (row)
- j represents the target position (column)
Attention Score: Value at (i,j) represents how much position i attends to position j. 


## Overview

The analysis pipeline:
1. Processes protein sequences through ESM2
2. Analyzes attention patterns around allosteric sites
3. Calculates sensitivity scores using random baselines
4. Identifies heads that show significant attention to allosteric regions

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/allostery_heads.git
cd allostery_heads

# Create a conda environment
conda create -n esm_env python=3.11
conda activate esm-env

# Install dependencies
pip install torch fair-esm numpy pandas tqdm
```

For M1 Macs you need PyTorch with MPS support instead of CUDA.

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"Is MPS (M1) available? {torch.backends.mps.is_available()}")
print(f"Is CUDA available? {torch.cuda.is_available()}")  # Should be False on M1
```

## Usage

### Basic Analysis

```python
from allosteric_analyzer import AllosticHeadAnalyzer

# Initialize analyzer
analyzer = AllosticHeadAnalyzer(threshold=0.3)

# Analyze protein sequence
sequence = "MPIMGSSVYITVELAIAVLAILGNVLVCWAV..."  # Your protein sequence
allosteric_sites = [85, 89, 246, 253]  # Known allosteric sites

# Get results
results = analyzer.analyze_protein(sequence, allosteric_sites)

# Access scores
impact_scores = results["impacts"].squeeze().tolist()
snr_values = results["snrs"].squeeze().tolist()
```

### Running A2A Receptor Analysis

```bash
python analyze_a2a.py
```

## Understanding the Scores

The analysis produces two key metrics:

1. **Impact Score**:
   - Ratio of attention to allosteric sites vs. random expectation
   - Score > 1.0: More attention to allosteric sites than random
   - Example: Score = 2.0 means twice as much attention to allosteric sites

2. **SNR (Signal-to-Noise Ratio)**:
   - Statistical significance of the impact score
   - Measures how many standard deviations above random
   - SNR > 2.0 typically considered significant

## Key Components

- `allosteric_analyzer.py`: Core class implementing analysis methods
- `analyze_a2a.py`: Example analysis of Adenosine A2A receptor

## Method Details

The analysis:
1. Extracts attention maps from ESM2
2. Calculates attention to allosteric sites
3. Compares with random baseline (1000 trials)
4. Excludes allosteric sites from random sampling
5. Computes impact scores and SNR values

## Example Output

```
Allosteric sensitivity analysis per attention head:
Head | Impact Score | SNR
-----------------------------------
   0 |       2.145 |   3.21
   1 |       1.876 |   2.98
...
  19 |       1.234 |   1.45

Most sensitive heads to allosteric sites [85, 89, 246, 253]:
(Impact > 1.5 and SNR > 2.0)
Head indices: [0, 1, 7, 14]
```

## Dependencies

- torch
- fair-esm
- numpy
- pandas
- tqdm

## References

1. Dong et al. (2024). Allo-Allo: Data-efficient prediction of allosteric sites. bioRxiv. DOI: pending
2. ESM2 Model: https://github.com/facebookresearch/esm

## Author

Aurelio A. Moya-García
