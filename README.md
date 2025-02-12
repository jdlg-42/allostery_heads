# allostery heads

* Actions are in `.actions`.
* Git commit templates are in `.commit_templates`.
* Use `prepare.exe` to prepare your migrations.
* Add big files with `git annex add` and regular files with `git add`.

Have fun!

Goal: 
Reproduce the paper Dong.2024.10.1101/2024.09.28.615583 Allo-Allo: Data efficient...
Get the allostery attention heads for a set of GPCRs.



## Setup
I work with Visual Studio Code and Copilot. The first thing to do is install the required libraries.

Create a Conda environment with Python 3.1. Python 3.12 (base environment might produce errors with some modules).


```
conda create -n esm_env python=3.11
conda activate esm_env
pip install torch
pip install tokenizers==0.20.3  # The version we know works
pip install esm  # This will get you ESM-C
```



### Test setup
Test, especially PyTorch. I need MPS support. M1 Macs don't work with CUDA.

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"Is MPS (M1) available? {torch.backends.mps.is_available()}")
print(f"Is CUDA available? {torch.cuda.is_available()}")  # Should be False on M1
```

## What info can we get from ESM-C?
`ESM-C_Att_map_explor.py` helps to import ESM-C 600m and to get to know what we can get from it. In particular, we want to get the attention maps (the actual numerical outputs and matrices produced by the attention heads).

The output:

```

Available attributes in output: ['__annotations__', '__attrs_attrs__', '__attrs_own_setattr__', '__class__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getstate__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__match_args__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__setstate__', '__sizeof__', '__slots__', '__str__', '__subclasshook__', '__weakref__', 'embeddings', 'hidden_states', 'logits', 'residue_annotation_logits']

```

This means that everything works with the attributes:

```

'embeddings'           # The token embeddings
'hidden_states'        # All intermediate layer representations
'logits'              # The final output logits
'residue_annotation_logits'  # Logits for residue annotations

```

The `hidden_states` look particularly promising for our purpose, as in transformer models like ESM-C, this typically contains the intermediate layer representations, including attention patterns.

## Allosteric Sensitivity
Attention maps in protein language models (PLMs) are matrices that represent the attention scores between different positions in a protein sequence. These scores indicate how much each position in the sequence should influence other positions, allowing the model to focus on specific relationships and patterns within the sequence.

In more detail:

- Attention Mechanism: The attention mechanism in PLMs computes a weighted sum of all positions in the sequence for each position. The attention mechanism determines the weights (attention scores) and indicate the importance of each position relative to others.

- Attention Heads: PLMs typically have multiple attention heads in each layer. Each attention head learns to focus on different aspects of the sequence, such as local structural elements or long-range interactions.

- Attention Maps: The attention scores for each head are organized into attention maps, which are tensors of shape [num_layers, num_heads, seq_len, seq_len]. Each element in the attention map represents the attention score between two positions in the sequence.

Attention maps can be used to interpret how the model understands the relationships between different amino acids in the protein sequence. For example, high attention scores between distant positions indicate long-range interactions important for the protein's function.

In the context of the AllosticHeadAnalyzer class, attention maps are used to identify which attention heads are most sensitive to allosteric sites, helping to understand how the model captures allosteric relationships in proteins.

`allosteric_sensitivity.py` Uses the class `AllosticHeadAnalyzer`. Using a class rather than several functions is more convenient because a class is a blueprint or template that bundles together related data and functions. It's like creating a custom tool that contains everything needed for a specific job. A class has several advantages over functions:
- Organization: Classes help keep related code together. In our case, everything related to protein analysis is in one place.
- Shared Resources: With classes, you can initialize things once and reuse them. Notice how in the functions approach, we need to load the model and tokenizer every time, while in the class approach, we do it once and keep using it.
- State Management: Classes can maintain state (remember information). In our analyzer, we want to keep using the same model and tokenizer rather than loading them repeatedly.
- Cleaner Code: Using a class makes the code cleaner to use.

### Initialize ESM-C
`analyzer = AllosticHeadAnalyzer(threshold=0.3)`, is actually doing two things:

1. Creating an instance (object) of the AllosticHeadAnalyzer class
2. Automatically calling the `__init__` function for that instance

The `__init__` method is a special method in Python classes called a constructor. It's automatically called when you create a new instance of the class. You don't need to explicitly call it because Python handles this for you.

Also, remember: When a variable is defined within a class's `__init__` method with `self., it becomes an instance variable that can be accessed by all other methods (functions) in the class.
```python
class AllosticHeadAnalyzer:
    def __init__(self):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        
    def some_other_method(self):
        # Can access self.device here because it's an instance variable
        print(f"Still using device: {self.device}")
```
        

### Getting attention Heads in ESM-C
Long story short. Neither ESM-C nor ESM3 provide info on the attention maps. I can get the embeddings and the hidden states, but not info on the attention heads.

At this point I can follow this strategy using ESM2, and develop a new strategy to get the "allostery scores" using ESM-C.

## Allosteric sensitivity with ESM2
The script `allosteric_sensitivity_ESM2.py` identifies the attention heads that focus in the allosteric sites of the protein.
### Getting the attention maps

```python
def get_attention_maps(self, sequence: str) -> torch.Tensor:
    # Tokenizes sequence and runs it through ESM2 model
    # Returns attention maps of shape [num_layers, num_heads, seq_len, seq_len]

```
 
Each attention map represents how much each position in the sequence attends to every other position. For each layer and head (ESM2 has 33 layers and 20 heads in each layer), you get a matrix where entry (i,j) shows how much position i pays attention to position j.

For each head, we have an attention matrix where:
- i represents the source position (row)
- j represents the target position (column)
Attention Score: Value at (i,j) represents how much position i attends to position j. 

### Computing allosteric impact
This is where the key calculation happens


```python

# Calculate w_allo (attention to allosteric sites)
w_allo = 0
mask = attention > self.threshold
for site in allo_sites:
    w_allo += torch.sum(attention[:, site][mask[:, site]]).item()    

# Calculate total activity w
w_total = torch.sum(attention > self.threshold).item()

# Calculate impact score p
impact = w_allo / w_total if w_total > 0 else 0


```

This implements two key equations from the paper:
- `w_allo`: Sum of attention scores above threshold for allosteric sites. For each allosteric site (j = site in allo_sites)
	- Look at all positions attending to it (column j in the matrix)
	- Sum up all attention values above threshold in that column

- `impact = w_allo / w_total`: Fraction of attention devoted to allosteric sites.

We measure hw strongly all positions in the protein attend to the allosteric site. This indicates which attention heads are sensitive to allostery.

### Interpreting the results.

```python
# Get scores for each head
impact_scores = results["impacts"].squeeze().tolist()

# Identify most sensitive heads (above mean)
mean_score = sum(impact_scores) / len(impact_scores)
sensitive_heads = [i for i, score in enumerate(impact_scores) if score > mean_score]

```

The output tells the impact score for each attention head (0.0 to 1.0), and which heads have above-average focus on allosteric sites.

For the adenosine a2a receptor with allosteric site extracted from the literature, I get:

```
Allosteric sensitivity scores per attention head:
Head 0: 0.008
Head 1: 0.006
Head 2: 0.009
Head 3: 0.008
Head 4: 0.007
Head 5: 0.008
Head 6: 0.008
Head 7: 0.008
Head 8: 0.006
Head 9: 0.008
Head 10: 0.008
Head 11: 0.007
Head 12: 0.008
Head 13: 0.007
Head 14: 0.007
Head 15: 0.007
Head 16: 0.005
Head 17: 0.006
Head 18: 0.008
Head 19: 0.008

Most sensitive heads to allosteric sites [85, 89, 246, 253]:
Head indices: [0, 2, 3, 5, 6, 7, 9, 10, 12, 15, 18, 19]

Sequence Length Analysis:
Sequence length: 316
Number of allosteric sites: 4
Ratio of sites to length: 0.013
Expected random attention: 0.003
Mean attention to allosteric sites: 0.003
Mean attention to non-allosteric sites: 0.003
Attention ratio (allosteric/non-allosteric): 1.000

```

Which are very low impacts. This might be caused by the sequence length. The impact scores are above random but may be diluted: in longer sequences, attention is distributed across more positions; and the same number of allosteric sites has less relative weight in longer sequences.

I add a normalization by sequence length

```python
def compute_allosteric_impact(self, attention_maps: List[torch.Tensor], allosteric_sites: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Modified impact calculation with length normalization"""
    # ...existing code...
    
    # Normalize impact by sequence length factor
    seq_len = attention_maps[0].size(-1)
    length_factor = (seq_len / len(allosteric_sites)) / 100  # Scaling factor
    impact = (w_allo / w_total) * length_factor if w_total > 0 else 0
    

```

Also add some statistics to analyze the sequence length effect. I get better sensitivity scores:

```
Allosteric sensitivity scores per attention head:
Head 0: 0.048
Head 1: 0.040
Head 2: 0.055
Head 3: 0.051
Head 4: 0.044
Head 5: 0.052
Head 6: 0.049
Head 7: 0.048
Head 8: 0.039
Head 9: 0.047
Head 10: 0.051
Head 11: 0.043
Head 12: 0.051
Head 13: 0.045
Head 14: 0.044
Head 15: 0.046
Head 16: 0.034
Head 17: 0.036
Head 18: 0.051
Head 19: 0.050

Most sensitive heads to allosteric sites [85, 89, 246, 253]:
Head indices: [0, 2, 3, 5, 6, 7, 9, 10, 12, 15, 18, 19]

Sequence Length Analysis:
Sequence length: 316
Number of allosteric sites: 4
Ratio of sites to length: 0.013
Expected random attention: 0.003
Mean attention to allosteric sites: 0.545
Mean attention to non-allosteric sites: 0.556
Attention ratio (allosteric/non-allosteric): 0.980
```

However, non-allosteric sites are receiving marginally more attention than allosteric sites. This could be due to several factors:
- Distance Effect: Allosteric sites might be far apart, making it harder for attention heads to capture their relationships. There are two types of distances: sequential distance—the number of residues between sites in the sequence; and structural distance—the physical distance in the 3D structure (which ESM2 tries to learn). The attention mechanism works by letting each position attend to all other positions, but attention tends to be stronger between nearby positions and therefore long-range relationships are harder to capture.
- Window Context: We're not considering the neighboring residues that might be part of the allosteric pathway
- Sequence Length: With 316 residues and only 4 allosteric sites, we need to consider local contexts.

I want to iddentify which residues attend preferentially to the allosteric site so I can look for the pathway between the allosteric site and those residues affected by modifications in the allosteric site. By definition the allosteric sites (i.e. those position in the allosteric pocket) will not be close in the sequence bt will be close in the structure, s I am not interested in extending the window context to include more positions in the allosteric sites because these sites are already defined and need to be a few amino acids.


I agree that attention decreases with the sequence distance. But I will use proteins with long sequences like GPCRs in which a few aminoacids form the allosteritc pocket i .e there are a few allosteric sites which are close in space but might be apart in the sequence.

I need to identify the allostery sensitive heads to then annotate each aminoacid of the protein with an "allostery score" which will be the average of the scores for that aminoacid in the allostery sensitive maps. This allostery score will be useful to identify the communication path between the allosteric sites and the orthosteric sites. That is I want to identify those amino acids that are attending to the allosteric sites. I want to do that by scoring the attention that each amino acid puts into the allosteric sites.

### Window Analysis Concept
The window analysis looks at attention patterns in a neighborhood around each allosteric site rather than just at the site itself. For example, with window_size=5, if we have an allosteric site at position 85, we analyze positions 80-90. This captures the local structural context that might be important for allosteric function.

Why It Matters for Allostery
In GPCRs and other proteins, allosteric sites don't work in isolation. A site at position 85 might work together with positions 83-87 to form a binding pocket or with positions 88-90 to transmit conformational changes. By analyzing attention patterns in windows, we can:

- Better identify heads that are sensitive to allosteric regions, not just individual positions
- Capture functional units like binding pockets or conformational hinges
- Reduce noise in the attention signals by considering local patterns
- Account for the fact that allosteric effects often propagate through local structural elements

```python

def analyze_window_scores(self, attention_maps: torch.Tensor, allosteric_sites: List[int], window_size: int = 5) -> Dict[int, float]:
        """
        Analyze attention patterns in windows around allosteric sites.
        Returns head scores based on attention to allosteric regions.
        
        Args:
            attention_maps: Attention tensors from model
            allosteric_sites: List of allosteric site positions (1-based)
            window_size: Size of window around each site (default: 5)
            
        Returns:
            Dictionary mapping head indices to window-based attention scores
        """
        head_scores = {}
        
        
        for head in range(self.num_heads):
            head_attention = attention_maps[0, head]
            total_score = 0
            valid_sites = 0
            
            for site in allosteric_sites:
                # Define window boundaries (convert to 0-based indexing)
                site_idx = site - 1
                start = max(0, site_idx - window_size)
                end = min(head_attention.shape[0], site_idx + window_size + 1)
                
                # Get average attention in window
                window_attention = head_attention[start:end, start:end]
                if window_attention.numel() > 0:  # Check if window is not empty
                    window_score = window_attention.mean().item()
                    if not np.isnan(window_score):  # Check for NaN
                        total_score += window_score
                        valid_sites += 1
            
            # Average across valid allosteric sites
            if valid_sites > 0:
                head_scores[head] = total_score / valid_sites
            else:
                head_scores[head] = 0.0
        
        return head_scores
    
    ```
    
    
    Currently I have this output:
    
    ```
    Window Analysis Results:
Head | Window Score | Visualization
--------------------------------------------------
   0 |  0.000000 | 
   1 |  0.000000 | 
   2 |  0.000000 | 
   3 |  0.000000 | 
   4 |  0.000000 | 
   5 |  0.000000 | 
   6 |  0.000000 | 
   7 |  0.000000 | 
   8 |  0.000000 | 
   9 |  0.000000 | 
  10 |  0.000000 | 
  11 |  0.000000 | 
  12 |  0.000000 | 
  13 |  0.000000 | 
  14 |  0.000000 | 
  15 |  0.000000 | 
  16 |  0.000000 | 
  17 |  0.000000 | 
  18 |  0.000000 | 
  19 |  0.000000 | 

Most sensitive heads (window analysis):
Head indices: []

Comparison of methods:
Original method sensitive heads: [0, 2, 3, 5, 6, 7, 9, 10, 12, 15, 18, 19]
Window method sensitive heads: []
Heads identified by both methods: set()

```

