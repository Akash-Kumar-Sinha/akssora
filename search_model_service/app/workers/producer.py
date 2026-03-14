import os, json, time
from confluent_kafka import Producer
from confluent_kafka.admin import AdminClient, NewTopic
from dotenv import load_dotenv

load_dotenv()

BOOTSTRAP_SERVERS = os.getenv("KAFKA_BROKER", "localhost:9092")
TOPIC_NAME = "video-processing"

def _wait_for_kafka(max_retries: int = 30, sleep_seconds: int = 2):
    admin = AdminClient({"bootstrap.servers": BOOTSTRAP_SERVERS})
    for attempt in range(1, max_retries + 1):
        try:
            admin.list_topics(timeout=5)
            print(f"[kafka] Broker reachable at {BOOTSTRAP_SERVERS}")
            return admin
        except Exception as exc:
            print(f"[kafka] Waiting for broker ({attempt}/{max_retries}): {exc}")
            time.sleep(sleep_seconds)
    raise RuntimeError(f"Kafka broker not reachable at {BOOTSTRAP_SERVERS}")


def ensure_topic():
    admin = _wait_for_kafka()
    metadata = admin.list_topics(timeout=10)
    if TOPIC_NAME in metadata.topics:
        print(f"[kafka] Topic already exists: {TOPIC_NAME}")
        return

    futures = admin.create_topics([NewTopic(TOPIC_NAME, num_partitions=1, replication_factor=1)])
    future = futures[TOPIC_NAME]
    try:
        future.result(15)
        print(f"[kafka] Topic created: {TOPIC_NAME}")
    except Exception as exc:
        print(f"[kafka] Topic create skipped/failed: {exc}")


def _get_producer() -> Producer:
    _wait_for_kafka()
    return Producer({"bootstrap.servers": BOOTSTRAP_SERVERS})

def publish_video_job(job: dict):
    producer = _get_producer()
    producer.produce(
        topic=TOPIC_NAME,
        key=job["job_id"],
        value=json.dumps(job).encode("utf-8")
    )
    producer.flush()