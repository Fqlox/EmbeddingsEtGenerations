# EmbeddingsEtGenerations
 
Repository comprenant deux scripts permettant d'élaborer une cartographie de représentation d'image de stable diffusion en cartographiant les images suivant l'encodages *CLIP* de leurs prompts.

`Tokenisation.ipynb` est un jupyter notebook qui permet de produire l'encodage clip puis sa réduction en 2 dimensions. Le script `makingimage.py` permet quant à lui de generer les images des prompts.

Le projet recquiert python, torch (cuda pour `makingimage.py`), transformers, diffusers, PIL.


Testé sous python 3.10.14 avec 
```
torch                     2.3.0
diffusers                 0.28.0
tokenizers                0.19.1
torch                     2.3.0
safetensors               0.4.3
xformers                  0.0.26.post1+cu118
```