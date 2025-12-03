from sentence_transformers import SentenceTransformer
import json

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "This is a video search engine.",
    "There is a man sitting on the top of the tree, eating a banana.",
    "It's a woman in a red dress walking on the beach.",
    ]


vectors = model.encode(sentences)

with open("text_embeddings/embeddings.txt", "w") as f:
    for text, vec in zip(sentences, vectors):
        record = {
            "text": text,
            "vector": vec.tolist()
        }
        f.write(json.dumps(record) + "\n")

print("Saved embeddings to embeddings.txt")
