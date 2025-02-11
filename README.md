# allostery heads

* Actions are in `.actions`.
* Git commit templates are in `.commit_templates`.
* Use `prepare.exe` to prepare your migrations.
* Add big files with `git annex add` and normal files with `git add`.

Have fun!

Goal: 
Reproduce the paper Dong.2024.10.1101/2024.09.28.615583 Allo-Allo: Data efficient...
Get the allostery attention heads for a set of GPCRs.



## Setup
I work with Visual Studio Code and Copilot. First thing is to install the required libraries.

Create a Conda environment with Python 3.1. Python 3.12 (base environment might produce errors with some modules).


```
conda create -n esm_env python=3.11
conda activate esm_env
pip install torch
pip install tokenizers==0.20.3  # The version we know works
pip install esm  # This will get you ESM-C
```



### Test setup
Test specially PyTorch. I need MPS support. M1 Macs don't work with CUDA.

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"Is MPS (M1) available? {torch.backends.mps.is_available()}")
print(f"Is CUDA available? {torch.cuda.is_available()}")  # Should be False on M1
```

## What info can we get from ESM-C?
`ESM-C_Att_map_explor.py` helps importing ESM-C 600m and getting to know what can we get from it. In particular we want to get the attention maps (the actual numerical outputs, matrices, that are produced by the attention heads).

The output:

```

Available attributes in output: ['__annotations__', '__attrs_attrs__', '__attrs_own_setattr__', '__class__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getstate__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__match_args__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__setstate__', '__sizeof__', '__slots__', '__str__', '__subclasshook__', '__weakref__', 'embeddings', 'hidden_states', 'logits', 'residue_annotation_logits']

```

Means that everything works with the attributes:

```

'embeddings'           # The token embeddings
'hidden_states'        # All intermediate layer representations
'logits'              # The final output logits
'residue_annotation_logits'  # Logits for residue annotations

```

The `hidden_states` looks particularly promising for our purpose, as in transformer models like ESM-C, this typically contains the intermediate layer representations including attention patterns.

## Allosteric Sensitivity
`allosteric_sensitivity.py` Uses the class `AllosticHeadAnalyzer`. It is more convenient to use a class rather than several functions because a class is kind of a blueprint or template that bundles together related data and functions. It's like creating a custom tool that contains everything needed for a specific job. A class has several advantaged over functions:
- Organization: Classes help keep related code together. In our case, everything related to protein analysis is in one place.
- Shared Resources: With classes, you can initialize things once and reuse them. Notice how in the functions approach, we need to load the model and tokenizer every time, while in the class approach we do it once and keep using it.
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
        





