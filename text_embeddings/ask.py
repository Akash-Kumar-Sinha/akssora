import os
import json
import ast
import numpy as np
from typing import List, Tuple, Dict, Any

from sentence_transformers import SentenceTransformer, util
import faiss



EMBEDDINGS_FILE = "./embeddings.txt"   

MODEL_NAME = "all-MiniLM-L6-v2"      

TOP_K = 5                            


def try_parse_json_line(line: str) -> Dict[str, Any]:
    try:
        return json.loads(line)
    except Exception:
        return None

def try_parse_python_literal(line: str) -> Any:
    try:
        return ast.literal_eval(line)
    except Exception:
        return None

def load_embeddings_from_file(path: str) -> Tuple[List[str], List[np.ndarray]]:
    """
    Tries multiple common formats. Returns (texts, vectors).
    Supported formats (per line):
      - JSON lines with {"id":..., "text":"...", "vector":[...]}
      - TSV/CSV with id \t text \t vector_json   (vector_json = "[1.0, 2.0, ...]")
      - plain lines with: text \t v1 v2 v3 ...
      - lines with: id \t v1 v2 v3 ...  (no text: text becomes id)
      - lines with: text<TAB>comma,separated,vector
      - If file contains only numeric vectors (no text), returns textual ids like "vec_0"
    """
    texts = []
    vectors = []

    with open(path, "r", encoding="utf-8") as f:
        for i, raw in enumerate(f):
            line = raw.strip()
            if not line:
                continue

            
            
            parsed = try_parse_json_line(line)
            if parsed:
                vec = parsed.get("vector") or parsed.get("embedding") or parsed.get("embeddings")
                if vec is None:
                    
                    
                    
                    
                    for v in parsed.values():
                        if isinstance(v, list) and all(isinstance(x, (int, float)) for x in v):
                            vec = v
                            break
                text = parsed.get("text") or parsed.get("content") or parsed.get("id") or parsed.get("doc") or parsed.get("title") or ""
                if vec:
                    texts.append(str(text) if text is not None else f"vec_{i}")
                    vectors.append(np.array(vec, dtype=np.float32))
                    continue

            
            
            parsed_py = try_parse_python_literal(line)
            if isinstance(parsed_py, (list, tuple)) and all(isinstance(x, (int, float)) for x in parsed_py):
                texts.append(f"vec_{i}")
                vectors.append(np.array(parsed_py, dtype=np.float32))
                continue

            
            
            parts = line.split("\t")
            if len(parts) >= 2:
                
                
                last = parts[-1].strip()
                vec_candidate = try_parse_python_literal(last)
                if isinstance(vec_candidate, (list, tuple)):
                    text = "\t".join(parts[:-1]).strip()
                    texts.append(text if text else f"vec_{i}")
                    vectors.append(np.array(vec_candidate, dtype=np.float32))
                    continue
                else:
                    
                    
                    nums = last.replace(",", " ").split()
                    if all(p.replace('.', '', 1).replace('-', '', 1).isdigit() for p in nums) and len(nums) > 5:
                        vec = [float(p) for p in nums]
                        text = "\t".join(parts[:-1]).strip()
                        texts.append(text if text else f"vec_{i}")
                        vectors.append(np.array(vec, dtype=np.float32))
                        continue

            
            
            parts_space = line.split()
            if len(parts_space) > 10 and all(p.replace('.', '', 1).replace('-', '', 1).isdigit() for p in parts_space[-len(parts_space)//2:]):
                
                
                
                
                j = len(parts_space)-1
                while j>=0 and parts_space[j].replace('.', '', 1).replace('-', '', 1).replace('e', '', 1).isdigit():
                    j -= 1
                j += 1
                text = " ".join(parts_space[:j])
                vec = [float(x) for x in parts_space[j:]]
                texts.append(text if text else f"vec_{i}")
                vectors.append(np.array(vec, dtype=np.float32))
                continue

            
            
            
            
            print(f"[WARN] Could not parse line {i+1}: {line[:80]}... (skipped)")

    if not vectors:
        raise ValueError("No vectors found in the file. Check the file format.")

    
    
    dims = set(v.shape[0] for v in vectors)
    if len(dims) != 1:
        raise ValueError(f"Inconsistent embedding dimensions found: {dims}")
    return texts, vectors



def build_faiss_index(vectors: List[np.ndarray], use_normalize: bool = True) -> Tuple[faiss.Index, np.ndarray]:
    """
    Builds a FAISS index (IndexFlatIP for cosine/dot) after optionally normalizing.
    Returns (index, numpy_matrix_vectors)
    """
    mat = np.vstack(vectors).astype('float32')
    if use_normalize:
        
        
        faiss.normalize_L2(mat)
    dim = mat.shape[1]

    
    
    index = faiss.IndexFlatIP(dim)
    index.add(mat)
    return index, mat



def retrieve(query: str, model: SentenceTransformer, index: faiss.Index, texts: List[str],
             mat_vectors: np.ndarray, top_k: int = 5, normalize_query: bool = True) -> List[Tuple[int, float, str]]:
    q_vec = model.encode(query, convert_to_numpy=True).astype('float32')
    if normalize_query:
        faiss.normalize_L2(q_vec.reshape(1, -1))
    D, I = index.search(q_vec.reshape(1, -1), top_k)  
    
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        results.append((int(idx), float(score), texts[idx]))
    return results



def make_answer_from_passages(query: str, hits: List[Tuple[int, float, str]], max_chars: int = 1000) -> str:
    """
    Basic extractive answer: concatenates top passages (deduped) and trims to max_chars.
    More advanced: run an LLM to synthesize.
    """
    seen = set()
    pieces = []
    for idx, score, text in hits:
        if text in seen:
            continue
        seen.add(text)
        pieces.append(f"[score={score:.3f}] {text}")
    joined = "\n\n".join(pieces)
    if len(joined) <= max_chars:
        return f"Query: {query}\n\nTop passages:\n\n{joined}"
    else:
        
        
        out = joined[:max_chars]
        out = out.rsplit("\n", 1)[0]
        return f"Query: {query}\n\nTop passages (truncated):\n\n{out}..."



def synthesize_with_openai(prompt_text: str, openai_api_key: str, model_name: str = "gpt-4o-mini") -> str:
    """
    Example placeholder showing how you'd call an LLM to synthesize an answer from retrieved context.
    This function is illustrative; implement using your favorite LLM client (openai, cohere, llama-cpp, etc.)
    """
    raise NotImplementedError("Implement this with your LLM provider of choice and an API key if desired.")



if __name__ == "__main__":
    if not os.path.exists(EMBEDDINGS_FILE):
        raise SystemExit(f"embeddings file not found: {EMBEDDINGS_FILE}")

    print("Loading embeddings from", EMBEDDINGS_FILE)
    texts, vectors = load_embeddings_from_file(EMBEDDINGS_FILE)
    print(f"Loaded {len(vectors)} vectors. Dimension = {vectors[0].shape[0]}")

    print("Building FAISS index (exact IP index).")
    index, mat = build_faiss_index(vectors, use_normalize=True)
    print("Index built. Total vectors indexed:", index.ntotal)

    print("Loading sentence-transformers model for query encoding:", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    
    
    print("\nReady. Type your question and press enter. Ctrl+C to exit.")
    try:
        while True:
            query = input("\nQuery> ").strip()
            if not query:
                print("Type a non-empty query.")
                continue

            hits = retrieve(query, model, index, texts, mat, top_k=TOP_K, normalize_query=True)
            if not hits:
                print("No results.")
                continue

            print("\nTop matches:")
            for rank, (idx, score, text) in enumerate(hits, start=1):
                print(f"{rank}. idx={idx}, score={score:.4f}")
                print(f"   {text[:400]}")
                print("")

            answer = make_answer_from_passages(query, hits, max_chars=1500)
            print("\n--- EXTRACTIVE ANSWER (from retrieved passages) ---\n")
            print(answer)

            
            
            
            
            
            

    except KeyboardInterrupt:
        print("\nExiting. Bye.")
