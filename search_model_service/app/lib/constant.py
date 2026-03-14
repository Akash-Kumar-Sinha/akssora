import os
from dotenv import load_dotenv
load_dotenv()
import boto3


def get_int_env(name: str) -> int:
	value = os.getenv(name)
	if value is None:
		raise ValueError(f"Missing required environment variable: {name}")
	return int(value)

MODEL_ID = os.getenv("MODEL_ID")
EMBEDDING_DIMENSION = get_int_env("EMBEDDING_DIMENSION")

TEXT_EMBEDDING_MODEL_ID = os.getenv("TEXT_EMBEDDING_MODEL_ID")
TEXT_INDEX_NAME= os.getenv("TEXT_INDEX_NAME")
TEXT_EMBEDDING_DIMENSION = get_int_env("TEXT_EMBEDDING_DIMENSION")

VECTOR_BUCKET = os.getenv("VECTOR_BUCKET")
INDEX_NAME = os.getenv("INDEX_NAME")

REGION=os.getenv("REGION")

S3_BUCKET = os.getenv("S3_BUCKET") 

S3_EMBEDDING_DESTINATION_URI = f"s3://{S3_BUCKET}/embeddings/"

s3_client = boto3.client("s3", region_name=REGION)

s3vector_client = boto3.client("s3vectors", region_name="us-east-1")

client = boto3.client("bedrock-runtime", region_name="us-east-1")

transcribe_client = boto3.client("transcribe", region_name=REGION)
