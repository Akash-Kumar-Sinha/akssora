import json
from concurrent.futures import ThreadPoolExecutor
from app.lib.generate_text_embedding import generate_text_embedding
from app.lib.constant import (
    MODEL_ID,
    EMBEDDING_DIMENSION,
    VECTOR_BUCKET,
    INDEX_NAME,
    TEXT_INDEX_NAME,
    s3vector_client,
    client,
)

RRF_K = 20
TOP_K_PER_QUERY = 25
RERANK_TOP_N = 30


def expand_query_multi(query: str) -> list[str]:
    response = client.invoke_model(
        body=json.dumps({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "text": f"""You are a visual search expert for a media search engine.
Given a user query, generate 4 search queries that together maximize recall across images and videos.

Return exactly 4 lines, each a different search angle:
a. A rich visual scene description (lighting, colors, setting, action)
2. The core subject/object isolated (e.g. "a man walking", "person at subway entrance")
3. The environment/context isolated (e.g. "metro station entrance, underground transit, subway exit")
4. A variation or related scene that might match (e.g. "commuter exiting underground train, man in subway corridor")

Rules:
- Each line is a standalone search query
- Focus on what is VISUALLY present
- No numbering, no labels, no bullet points - just 4 plain lines
- Max 2 sentences per line

User query: {query}"""
                        }
                    ]
                }
            ],
            "inferenceConfig": {
                "maxTokens": 300,
                "temperature": 0.4
            }
        }),
        modelId="amazon.nova-lite-v1:0",
        accept="application/json",
        contentType="application/json"
    )
    body = json.loads(response["body"].read())
    raw = body["output"]["message"]["content"][0]["text"].strip()
    queries = [q.strip() for q in raw.split("\n") if q.strip()][:4]
    return queries


def generate_visual_query_embedding(text: str) -> list:
    response = client.invoke_model(
        body=json.dumps({
            "taskType": "SINGLE_EMBEDDING",
            "singleEmbeddingParams": {
                "embeddingPurpose": "GENERIC_INDEX",
                "embeddingDimension": EMBEDDING_DIMENSION,
                "text": {
                    "truncationMode": "END",
                    "value": text
                }
            }
        }),
        modelId=MODEL_ID,
        accept="application/json",
        contentType="application/json"
    )
    body = json.loads(response["body"].read())
    return body["embeddings"][0]["embedding"]


def query_visual_index(embedding: list, top_k: int = TOP_K_PER_QUERY) -> list:
    response = s3vector_client.query_vectors(
        vectorBucketName=VECTOR_BUCKET,
        indexName=INDEX_NAME,
        queryVector={"float32": embedding},
        topK=top_k,
        returnDistance=True,
        returnMetadata=True
    )
    return response["vectors"]


def query_text_index(embedding: list, top_k: int = TOP_K_PER_QUERY) -> list:
    response = s3vector_client.query_vectors(
        vectorBucketName=VECTOR_BUCKET,
        indexName=TEXT_INDEX_NAME,
        queryVector={"float32": embedding},
        topK=top_k,
        returnDistance=True,
        returnMetadata=True
    )
    return response["vectors"]


def reciprocal_rank_fusion(*result_lists, k: int = RRF_K) -> list:
    def get_dedup_key(vector):
        meta = vector.get("metadata", {})
        return meta.get("segment_id") or meta.get("image_id") or vector.get("key", "")

    scores = {}
    records = {}

    for result_list in result_lists:
        for rank, vector in enumerate(result_list):
            key = get_dedup_key(vector)
            scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
            if key not in records or vector["distance"] < records[key]["distance"]:
                records[key] = vector

    merged = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
    return [(records[k], scores[k]) for k in merged]


def rerank_results(query: str, candidates: list) -> list:
    if not candidates:
        return candidates

    candidate_lines = []
    for i, (vector, _) in enumerate(candidates):
        meta = vector["metadata"]
        description = meta.get("description", "")
        transcript = meta.get("transcript", "")
        media_type = meta.get("type", "unknown")
        text_hint = description or transcript or ""
        candidate_lines.append(f"{i}: [{media_type}] {text_hint[:150]}")

    candidates_text = "\n".join(candidate_lines)

    response = client.invoke_model(
        body=json.dumps({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "text": f"""You are a search relevance judge.
Given a user query and a list of media candidates (images/videos described by their metadata), score each candidate from 0–10 based on how likely it visually matches the query.

Query: "{query}"

Candidates:
{candidates_text}

Return ONLY a JSON array of scores in order, one number per candidate, e.g. [8, 3, 9, 1, ...]
No explanation, no markdown, just the JSON array."""
                        }
                    ]
                }
            ],
            "inferenceConfig": {
                "maxTokens": 200,
                "temperature": 0.0
            }
        }),
        modelId="amazon.nova-lite-v1:0",
        accept="application/json",
        contentType="application/json"
    )
    body = json.loads(response["body"].read())
    raw = body["output"]["message"]["content"][0]["text"].strip()

    try:
        scores = json.loads(raw)
        if not isinstance(scores, list) or len(scores) != len(candidates):
            raise ValueError("Score count mismatch")
    except Exception as e:
        return candidates

    reranked = [
        (vector, float(score) / 10.0)
        for (vector, _), score in zip(candidates, scores)
    ]
    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked


def search(query: str):
    with ThreadPoolExecutor(max_workers=2) as executor:
        multi_future = executor.submit(expand_query_multi, query)
        text_embed_future = executor.submit(generate_text_embedding, query)
        expanded_queries = multi_future.result()
        text_embedding = text_embed_future.result()

    with ThreadPoolExecutor(max_workers=len(expanded_queries)) as executor:
        visual_embeddings = [
            f.result() for f in [
                executor.submit(generate_visual_query_embedding, q)
                for q in expanded_queries
            ]
        ]

    with ThreadPoolExecutor(max_workers=len(visual_embeddings) + 1) as executor:
        visual_futures = [
            executor.submit(query_visual_index, emb, TOP_K_PER_QUERY)
            for emb in visual_embeddings
        ]
        text_future = executor.submit(query_text_index, text_embedding, TOP_K_PER_QUERY)
        all_visual_results = [f.result() for f in visual_futures]
        text_results = text_future.result()

    merged = reciprocal_rank_fusion(*all_visual_results, text_results)

    top_candidates = merged[:RERANK_TOP_N]
    reranked = rerank_results(query, top_candidates)

    video_map = {}
    image_results = []

    for vector, final_score in reranked:
        metadata = vector["metadata"]
        result_type = metadata.get("type")

        if result_type == "image":
            image_results.append({
                "type": "image",
                "url": metadata.get("url"),
                "score": final_score
            })

        elif result_type == "video":
            url = metadata.get("url")
            if url not in video_map:
                video_map[url] = {
                    "type": "video",
                    "url": url,
                    "score": final_score,
                    "segments": []
                }
            video_map[url]["segments"].append({
                "start_time": metadata.get("start_time"),
                "end_time": metadata.get("end_time"),
                "transcript": metadata.get("transcript"),
                "score": final_score
            })
            video_map[url]["segments"].sort(key=lambda x: x["score"], reverse=True)

    results = image_results + list(video_map.values())
    results.sort(key=lambda x: x["score"], reverse=True)
    return results