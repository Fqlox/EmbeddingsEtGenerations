"""
Script pour la génération des différentes images
"""


from tqdm import tqdm

# Difffusers library
# On utilise StableDiffusionXLPipeline car nous utilisons le modèle RealitiesEdgeXLLIGHTNING_LIGHTNING34Step
from diffusers import  StableDiffusionXLPipeline, DDIMScheduler
import torch

from uuid import uuid4
import json


# Batch size pour que le modèle génére quatre image par quatre image ici. Le batch size est dependant du gpu
BATCH_SIZE = 4
saving_data = []
# Repertoire où sera enregistrer les images
dir_name = "image"



# On produit des prompts exactement de la même manière que pour le PCA => voir tokenization.ipnyb
def pipe_to_list(pipe:str):
    return pipe.split("|")

subjects = "A man | a woman | landscape forest | view from the exterior of a cavern | Beautiful landscape, sunset | Montain landscape | city seen from above the ground, skyscraper |   a person with a beautiful smile | crowd of people on the street | Small dog on a couch | a cat next to a window | a space alien "
lights = "Natural light | hard light | soft light "
photographics = "Intricate detail, ultra sharp | 50mm | 120mm | long shot | ISO 1000 | ISO 100 | long pose shot | portrait shot | close-up | ultra close-up | long distance shoot | fisheye | 8mm | 16m"

prompts = [pipe_to_list(subjects), pipe_to_list(lights), pipe_to_list(photographics)]

from itertools import product


prompts_list = []
for item in product(*prompts):
    prompts_list.append(", ".join(item))

print(f"Nombre de prompts: {len(prompts_list)}")

# Chemin absolue vers le modèle (ici RealitiesEdgeXLLIGHTNING_LIGHTNING34Step) 
# On pourrait aussi le telecharger depuis HG_HUB avec la méthode "from_pretrained" mais dans notre cas nous avons déjà les modèles en format safetensors pour produire des inférences sur automatic1111 et comfyui
diffusion_model_path = "path_to_model/RealitiesEdgeXLLIGHTNING_LIGHTNING34Step.safetensors"

pipe = StableDiffusionXLPipeline.from_single_file(f"{diffusion_model_path}", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")

# Boucle itérative générant toutes les images suivant les prompts
for i in tqdm(range(int(len(prompts_list)/BATCH_SIZE))):
    selected = prompts_list[i*BATCH_SIZE:(i*BATCH_SIZE+BATCH_SIZE)]
    print(selected)
    image_name = [f"{dir_name}/{uuid4()}.jpg" for i in range(BATCH_SIZE)]

    images = pipe(prompt=selected,
            width=1024,
            height=1024,
            num_inference_steps=6,
            guidance=4
    ).images

    for index, prompt in enumerate(selected):
        
        saving_data.append({
            "prompt" : prompt,
            "image" : image_name[index]
        })
        # Enregistrement des images
        images[index].save(f"{image_name[index]}")

# On enregistre la liste des associations entre prompt et nom des fichiers images :
#   [{"prompt": "Lorem Ipsum", "image" : "path/to/image.jpg"}, {...}]
with open("saving.json", "w+", encoding="UTF-8") as f:
    json.dump(saving_data, f, ensure_ascii=True, indent=1)