import os
import torch
import numpy as np
import open_clip
from PIL import Image

FRAMES_DIR = "frames"
SAVE_FILE = "embeddings.txt"


model, preprocess, _ = open_clip.create_model_and_transforms(
    model_name="ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)
model.eval()  

def get_image_embedding(path):
    image = preprocess(Image.open(path)).unsqueeze(0)

    with torch.no_grad():
        emb = model.encode_image(image)

    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.squeeze().numpy()

def main():
    with open(SAVE_FILE, "w") as f:
        for file_name in sorted(os.listdir(FRAMES_DIR)):
            if not file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            path = os.path.join(FRAMES_DIR, file_name)
            emb = get_image_embedding(path)

            vector_str = " ".join(map(str, emb.tolist()))
            f.write(f"{file_name} :: {vector_str}\n")

            print("Saved:", file_name)

if __name__ == "__main__":
    main()
