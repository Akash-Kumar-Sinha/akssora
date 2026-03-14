import os, json, time
from confluent_kafka import Consumer
from app.content_handler.video import process_video
from app.workers.producer import _wait_for_kafka, TOPIC_NAME

def consumer():
    _wait_for_kafka()

    consumer = Consumer({
        "bootstrap.servers": os.getenv("KAFKA_BROKER", "localhost:9092"),
        "group.id": "video-workers",
        "auto.offset.reset": "earliest"
    })
    consumer.subscribe([TOPIC_NAME])

    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                time.sleep(0.1)
                continue
            if msg.error():
                print("[consumer] Poll error:", msg.error())
                continue

            job = json.loads(msg.value().decode("utf-8"))

            process_video(job["s3_url"], job["url"])

    finally:
        consumer.close()