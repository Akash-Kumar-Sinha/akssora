import os
import torch
import numpy as np
from PIL import Image
import open_clip

EMB_FILE = "embeddings.txt"
FRAMES_DIR = "frames"

model, preprocess, _ = open_clip.create_model_and_transforms(
    model_name="ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)
model.eval()

tokenizer = open_clip.get_tokenizer("ViT-B-32")

def load_embeddings():
    data = []
    with open(EMB_FILE, "r") as f:
        for line in f:
            name, vec = line.split(" :: ")
            vector = np.array(list(map(float, vec.split())))
            data.append((name, vector))
    return data

def embed_query(text):
    tokens = tokenizer([text])  

    with torch.no_grad():
        emb = model.encode_text(tokens)

    emb = emb.squeeze().numpy()
    emb = emb / np.linalg.norm(emb)
    return emb

def search_best_matches(query, images, top_k=3):
    q = embed_query(query)

    scores = []

    for name, vec in images:
        score = np.dot(q, vec)
        scores.append((name, score))

    # Sort by score (descending)
    scores.sort(key=lambda x: x[1], reverse=True)

    return scores[:top_k]


def show_image(image_path):
    img = Image.open(image_path)
    img.show()

if __name__ == "__main__":
    images = load_embeddings()
    print("Loaded", len(images), "embeddings.")

    query = input("Ask a question: ")

    top_matches = search_best_matches(query, images, top_k=3)

    for name, score in top_matches:
        print("Match:", name, "| score:", score)
        show_image(os.path.join(FRAMES_DIR, name))


    # show_image(os.path.join(FRAMES_DIR, best))
