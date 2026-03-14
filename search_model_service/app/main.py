from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.content_handler.routes import router as videos_router
from app.search_handler.search import search
from contextlib import asynccontextmanager
from app.workers.consumer import consumer
from app.workers.producer import ensure_topic
import threading

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[startup] Ensuring Kafka topic exists")
    ensure_topic()
    thread = threading.Thread(target=consumer, daemon=True)
    thread.start()
    print("[startup] Kafka consumer started")
    yield

app = FastAPI(lifespan=lifespan)

origins = [
    "http://localhost.8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(videos_router)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/search")
def search_end(query: str):
    return search(query)