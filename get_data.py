from datasets import load_dataset
import os
from tqdm import tqdm
import requests
from PIL import Image
from io import BytesIO

ds = load_dataset("huggan/wikiart", split="train")

os.makedirs("input_photos/cubism", exist_ok=True)

cubism_count = 1

for example in tqdm(ds, desc="Saving Cubism images"):
    if example['style'] == 7 or example['style'] == 'Cubism':
        image = example['image'] 
        filename = f"input_photos/cubism/{cubism_count}.jpg"
        image.save(filename)
        print(f"Saved: {filename}")
        cubism_count += 1


os.makedirs("input_photos/nature", exist_ok=True)

ds = load_dataset("mertcobanov/nature-dataset", split="train")

nature_count = 1

for example in tqdm(ds, desc="Saving Nature images"):
    if nature_count < 2236:
        image = example['image']
        filename = f"input_photos/nature/{nature_count}.jpg"
        image.save(filename)
        print(f"Saved: {filename}")
        nature_count += 1
    else:
        break