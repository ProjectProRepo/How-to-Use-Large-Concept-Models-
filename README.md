# LCM Text Generation Model

## Overview
This repository implements a Large Concept Model (LCM) for text generation using Transformer-based embeddings. The model leverages **SONAR embeddings**, **WTPSplit**, and **Torch Transformers** to break text into conceptual chunks, encode them, transform the embeddings, and generate new concepts iteratively.

## Features
- **Text-to-Vector Embedding**: Converts input text into SONAR embeddings.
- **Transformer-Based Concept Generation**: Uses a Transformer model to refine embeddings.
- **Vector-to-Text Generation**: Decodes transformed embeddings back into meaningful text.
- **Supports CUDA**: Runs on GPU for faster inference.

## Installation
Before running the model, install the required dependencies:

```sh
pip install -q fairseq2==v0.3.0rc1 --pre --extra-index-url  https://fair.pkg.atmeta.com/fairseq2/whl/rc/pt2.5.1/cu124 --upgrade
pip install -q sonar-space
pip install -q wtpsplit sonar
```

## Usage
To run the model, initialize the configuration and generate output as follows:

```python
import torch
from your_module import LCMModel, LCMConfig

# Ensure correct device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Initialize model
config = LCMConfig()
lcm = LCMModel(config)

# Run text generation
text = "This is a test sentence."
output = lcm.generate(text, num_generated_concepts=2)
print("\nGenerated Output:", output)
```

## Expected Output
The model will generate a sequence of conceptually extended text based on the input:

```
Generated Output: This is a test sentence. In the meantime, I'm going to take a look at some of the things that I've learned in the past...
```

## Notes
- If you face CUDA-related issues, set the runtime to GPU when using the Google colab environment.
- For learning about LCMs from scratch and a detailed explanation of the code in this repo, please read the article: [Large Concept Models by ProjectPro](https://www.projectpro.io/article/large-concept-models/1114)

## Citation
If you use LCM-torch in your research or projects, please cite the original Large Concept Model paper:
```
@article{lcm2024,
  author = {{LCM team}, Lo\"{i}c Barrault, Paul-Ambroise Duquenne, Maha Elbayad, Artyom Kozhevnikov, Belen Alastruey, Pierre Andrews, Mariano Coria, Guillaume Couairon, Marta R. Costa-juss\`{a}, David Dale, Hady Elsahar, Kevin Heffernan, Jo\~{a}o Maria Janeiro, Tuan Tran, Christophe Ropers, Eduardo Sánchez, Robin San Roman, Alexandre Mourachko, Safiyyah Saleem, Holger Schwenk},
  title = {{Large Concept Models}: Language Modeling in a Sentence Representation Space},
  publisher = {arXiv},
  year = {2024},
  url = {https://arxiv.org/abs/2412.08821},
}
```
