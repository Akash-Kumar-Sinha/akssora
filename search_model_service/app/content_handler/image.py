import os
import mimetypes
import json
import base64
import uuid
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor

from app.lib.constant import (
    MODEL_ID,
    EMBEDDING_DIMENSION,
    VECTOR_BUCKET,
    INDEX_NAME,
    s3vector_client,
    s3_client,
    client,
    S3_BUCKET,
    TEXT_INDEX_NAME
) 


from app.lib.file_name import file_name

from app.lib.generate_text_embedding import generate_text_embedding


def _resolve_image_format(content_type: str | None, file_path: str) -> str:
    if content_type:
        mime = content_type.lower().strip()
    else:
        mime = (mimetypes.guess_type(file_path)[0] or "image/png").lower()

    mime_to_format = {
        "image/png": "png",
        "image/jpeg": "jpeg",
        "image/jpg": "jpeg",
        "image/webp": "webp",
        "image/gif": "gif",
    }
    return mime_to_format.get(mime, "png")

def upload_image_to_s3(image_path):
    try:
        object_key = os.path.basename(image_path)
        content_type, _ = mimetypes.guess_type(image_path)
        s3_key = f"images/{object_key}"

        s3_client.upload_file(
            image_path,
            S3_BUCKET,
            s3_key,
            ExtraArgs={
                "ContentType": content_type or "image/png"}  
        )   

        image_url = f"https://{S3_BUCKET}.s3.amazonaws.com/{s3_key}"

        return image_url

    except ClientError as e:
        print("S3 upload failed:", e)
        return None

    except Exception as e:
        print("Unexpected error:", e)
        return None



def generate_image_description(image_bytes: bytes, image_format: str) -> str:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    result = client.invoke_model(
        body=json.dumps({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "image": {
                                "format": image_format,
                                "source": {"bytes": encoded}
                            }
                        },
                        {
                            "text": "Describe this image in detail - objects, people, setting, colors, actions, composition. Be concise, max 3 sentences."
                        }
                    ]
                }
            ],
            "inferenceConfig": {
                "maxTokens": 200,
                "temperature": 0.3
            }
        }),
        modelId="amazon.nova-lite-v1:0",
        accept="application/json",
        contentType="application/json"
    )
    body = json.loads(result["body"].read())
    return body["output"]["message"]["content"][0]["text"].strip()


def generate_image_embeddings(image_bytes: bytes, image_format: str):
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    request_body = {
        "taskType": "SINGLE_EMBEDDING",
        "singleEmbeddingParams": {
            "embeddingPurpose": "GENERIC_INDEX",
            "embeddingDimension": EMBEDDING_DIMENSION,
            "image": {
                "format": image_format,
                "source": {"bytes": encoded}
            },
        },
    }
    response = client.invoke_model(
        body=json.dumps(request_body),
        modelId=MODEL_ID,
        contentType="application/json",
    )
    response_body = json.loads(response["body"].read())
    return response_body["embeddings"][0]["embedding"]


def save_embedding_to_vector_db(embedding, text_embedding, image_url: str, description: str):
    image_id = str(uuid.uuid4())
    shared_metadata = {
        "type": "image",
        "url": image_url,
        "description": description,
        "image_id": image_id
    }

    s3vector_client.put_vectors(
        vectorBucketName=VECTOR_BUCKET,
        indexName=INDEX_NAME,
        vectors=[{
            "key": f"{image_id}#visual",
            "data": {"float32": embedding},
            "metadata": {**shared_metadata, "embedding_type": "visual"}
        }]
    )

    if text_embedding:
        s3vector_client.put_vectors(
            vectorBucketName=VECTOR_BUCKET,
            indexName=TEXT_INDEX_NAME,
            vectors=[{
                "key": f"{image_id}#text",
                "data": {"float32": text_embedding},
                "metadata": {**shared_metadata, "embedding_type": "text"}
            }]
        )


def upload_image(file):
    try:
        file_location = file_name(file)
        with open(file_location, "wb") as buffer:
            buffer.write(file.file.read())

        with open(file_location, "rb") as source:
            raw_image_bytes = source.read()

        image_format = _resolve_image_format(file.content_type, file_location)

        image_url = upload_image_to_s3(file_location)
        if image_url is None:
            print("[image] Skipping image due to upload failure")
            return

        description = generate_image_description(raw_image_bytes, image_format)

        with ThreadPoolExecutor(max_workers=2) as executor:
            visual_future = executor.submit(generate_image_embeddings, raw_image_bytes, image_format)
            text_future = executor.submit(generate_text_embedding, description)
            visual_embedding = visual_future.result()
            text_embedding = text_future.result()

        save_embedding_to_vector_db(visual_embedding, text_embedding, image_url, description)

        os.remove(file_location)
        return image_url

    except Exception as e:
        print("[image] Error processing uploaded image:", e)
        return None