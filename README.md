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

Create a Conda environment with Python 3.9. Python 3.12 (base environment might produce errors with some modules)

`conda create --name plm python=3.9
conda activate plm && python --version`  

`# Install PyTorch with M1 support
conda install pytorch torchvision -c pytorch-nightly`

`# Install other dependencies
conda install pandas numpy tqdm
conda install -c conda-forge transformers`

`conda install jupyter`

`conda install scikit-learn  # For later if you want to implement the RF classifier
conda install matplotlib    # For visualizing results`

### Test setup
After installation, you can verify everything is working by running Python and importing the modules.
```python
import torch
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import numpy as np
from tqdm import tqdm

# Verify PyTorch is using MPS (Apple's Metal Performance Shaders)
print(f"Is MPS (Metal Performance Shaders) available: {torch.backends.mps.is_available()}")
``



