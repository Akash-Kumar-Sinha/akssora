import uuid
from concurrent.futures import ThreadPoolExecutor
import urllib.request
import time
import os
import mimetypes
import json
from botocore.exceptions import ClientError

from app.lib.constant import (
    MODEL_ID, 
    EMBEDDING_DIMENSION,
    VECTOR_BUCKET, 
    INDEX_NAME, 
    s3vector_client,
    s3_client,
    client,
    S3_BUCKET,
    S3_EMBEDDING_DESTINATION_URI,
    transcribe_client, 
    TEXT_INDEX_NAME
)

from app.lib.generate_text_embedding import generate_text_embedding

def upload_video_to_s3(video_path):
    try:
        print(f"[video] Uploading file to S3: {video_path}")
        object_key = os.path.basename(video_path)
        content_type, _ = mimetypes.guess_type(video_path)

        s3_key = f"videos/{object_key}"

        s3_client.upload_file(
            video_path,
            S3_BUCKET,
            s3_key,
            ExtraArgs={
                "ContentType": content_type or "video/mp4"
            }
        )

        s3_uri = f"s3://{S3_BUCKET}/{s3_key}"
        url = f"https://{S3_BUCKET}.s3.ap-south-1.amazonaws.com/{s3_key}"

        return {
            "s3_url": s3_uri,
            "url":url
        }

    except ClientError as e:
        print("[video] S3 upload failed:", e)
        return None

    except Exception as e:
        print("[video] Unexpected upload error:", e)
        return None

def transcribe_video(video_path, job_name):
    transcribe_client.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={"MediaFileUri": video_path},
        MediaFormat="mp4",
        LanguageCode="en-US",
        Settings={"ShowSpeakerLabels": False}
    )

    while True:
        job = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
        status = job["TranscriptionJob"]["TranscriptionJobStatus"]
        if status in ["COMPLETED", "FAILED"]:
            break
        time.sleep(10)

    if status == "FAILED":
        print(f"[video] Transcription job failed: {job_name}")
        return None

    transcript_uri = job["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
    with urllib.request.urlopen(transcript_uri) as f:
        transcript_data = json.loads(f.read().decode("utf-8"))

    return transcript_data["results"]["items"]


def extract_transcript_for_segment(items, start_time, end_time):
    if not items:
        return ""

    words = []
    for item in items:
        if item["type"] != "pronunciation":
            continue
        word_start = float(item.get("start_time", 0))
        word_end = float(item.get("end_time", 0))

        if word_start >= start_time and word_end <= end_time:
            words.append(item["alternatives"][0]["content"])

    return " ".join(words)

def generate_video_embedding(video_path):
    segment_duration = 15
    segment_step = 10
    job_name = f"ghost-editor-{int(time.time())}"

    transcript_items = transcribe_video(video_path, job_name)
    is_muted = transcript_items is None

    embedding_mode = "VIDEO_ONLY" if is_muted else "AUDIO_VIDEO_COMBINED"
    embeddings_file_key = "embedding-video.jsonl" if is_muted else "embedding-audio-video.jsonl"

    model_input = {
        "taskType": "SEGMENTED_EMBEDDING",
        "segmentedEmbeddingParams": {
            "embeddingPurpose": "GENERIC_INDEX",
            "embeddingDimension": EMBEDDING_DIMENSION,
            "video": {
                "format": "mp4",
                "embeddingMode": embedding_mode,
                "source": {
                    "s3Location": {"uri": video_path}
                },
                "segmentationConfig": {
                    "durationSeconds": segment_duration
                }
            }
        }
    }

    response = client.start_async_invoke(
        modelId=MODEL_ID,
        modelInput=model_input,
        outputDataConfig={
            "s3OutputDataConfig": {
                "s3Uri": S3_EMBEDDING_DESTINATION_URI
            }
        }
    )
    invocation_arn = response["invocationArn"]

    while True:
        job = client.get_async_invoke(invocationArn=invocation_arn)
        status = job["status"]
        if status != "InProgress":
            break
        time.sleep(10)
    if status != "Completed":
        raise Exception("Embedding job failed")

    output_s3_uri = job["outputDataConfig"]["s3OutputDataConfig"]["s3Uri"]
    s3_uri_parts = output_s3_uri.replace("s3://", "").split("/", 1)
    bucket = s3_uri_parts[0]
    prefix = s3_uri_parts[1]
    embeddings_key = f"{prefix}/{embeddings_file_key}"
    response = s3_client.get_object(Bucket=bucket, Key=embeddings_key)
    content = response["Body"].read().decode("utf-8")

    raw_segments = content.strip().split("\n")
    embeddings = []
    total_duration = len(raw_segments) * segment_duration

    window_start = 0
    while window_start < total_duration:
        window_end = window_start + segment_duration

        overlapping_embeddings = []
        for i, line in enumerate(raw_segments):
            raw_start = i * segment_duration
            raw_end = raw_start + segment_duration
            if raw_start < window_end and raw_end > window_start:
                data = json.loads(line)
                overlapping_embeddings.append(data["embedding"])

        if overlapping_embeddings:
            avg_embedding = [
                sum(dim) / len(overlapping_embeddings)
                for dim in zip(*overlapping_embeddings)
            ]
            transcript = extract_transcript_for_segment(transcript_items, window_start, window_end)

            text_embedding = None
            if transcript.strip():
                with ThreadPoolExecutor(max_workers=1) as executor:
                    text_future = executor.submit(generate_text_embedding, transcript)
                    text_embedding = text_future.result()

            embeddings.append({
                "start_time": window_start,
                "end_time": window_end,
                "embedding": avg_embedding,
                "transcript": transcript,
                "text_embedding": text_embedding,
            })

        window_start += segment_step

    return embeddings


def save_video_embedding_to_vector_db(embeddings, s3_uri, url):
    visual_vectors = []
    text_vectors = []

    for segment in embeddings:
        start_time = float(segment.get("start_time", 0.0))
        end_time = float(segment.get("end_time", 0.0))
        visual_embedding = segment.get("embedding")
        text_embedding = segment.get("text_embedding")
        transcript = segment.get("transcript", "")

        if not visual_embedding:
            continue

        segment_id = str(uuid.uuid4())
        shared_metadata = {
            "type": "video",
            "s3_url": s3_uri,
            "url": url,
            "start_time": start_time,
            "end_time": end_time,
            "transcript": transcript,
            "segment_id": segment_id
        }

        visual_vectors.append({
            "key": f"{segment_id}#visual",
            "data": {"float32": visual_embedding},
            "metadata": {**shared_metadata, "embedding_type": "visual"}
        })

        if text_embedding:
            text_vectors.append({
                "key": f"{segment_id}#text",
                "data": {"float32": text_embedding},
                "metadata": {**shared_metadata, "embedding_type": "text"}
            })

    if visual_vectors:
        s3vector_client.put_vectors(
            vectorBucketName=VECTOR_BUCKET,
            indexName=INDEX_NAME,     
            vectors=visual_vectors
        )

    if text_vectors:
        s3vector_client.put_vectors(
            vectorBucketName=VECTOR_BUCKET,
            indexName=TEXT_INDEX_NAME,
            vectors=text_vectors
        )

def process_video(s3_uri, url):
    embeddings = generate_video_embedding(s3_uri)
    save_video_embedding_to_vector_db(embeddings, s3_uri, url)
    print(f"[video] Processing completed")