import logging
from PIL import Image
from insightface.app import FaceAnalysis
from pydantic import BaseModel
import os
from datetime import datetime
import numpy as np
import io

from dotenv import load_dotenv
load_dotenv()




logger = logging.getLogger(__name__)

# ── MongoDB ────────────────────────────────────────────────────────────────
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = os.getenv("MONGODB_URI")

print(uri)

client = MongoClient(uri, server_api=ServerApi('1'))

try:
    client.admin.command('ping')
    logger.info("Connected to MongoDB!")
except Exception as e:
    logger.error(f"MongoDB connection failed: {e}")

database_name = "attendance_system"
collection_name = "dbms"
collection = client[database_name][collection_name]

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

print("SUPABASE_URL:", SUPABASE_URL)
print("SUPABASE_KEY:", SUPABASE_KEY)

# ── Supabase ───────────────────────────────────────────────────────────────
from supabase import create_client, Client




supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY"),
)

SUPABASE_BUCKET = "attendance_metadata"


# ── MetaDataDB ─────────────────────────────────────────────────────────────

class MetaDataDB:
    def __init__(self):
        pass

    # ── Image: Supabase Storage ────────────────────────────────────────────
    def save_image(self, image_key: str, image) -> str:
        """Upload image to Supabase Storage, return public URL."""
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG")
        buf.seek(0)

        supabase.storage.from_(SUPABASE_BUCKET).upload(
            image_key,
            buf.read(),
            file_options={"content-type": "image/jpeg", "upsert": "true"},
        )

        public_url = supabase.storage.from_(SUPABASE_BUCKET).get_public_url(image_key)
        logger.info(f"Image uploaded to Supabase → {public_url}")
        return public_url

    def load_image(self, image_key: str) -> np.ndarray:
        """Download image from Supabase Storage, return numpy array."""
        raw = supabase.storage.from_(SUPABASE_BUCKET).download(image_key)
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        logger.info(f"Image loaded from Supabase → {image_key}")
        return np.array(image)

    # ── Metadata + embedding: MongoDB ─────────────────────────────────────
    def save_metadata(self, record: dict) -> str:
        """Insert a single face record (with embedding) into MongoDB."""
        result = collection.insert_one(record)
        logger.info(f"Metadata saved to MongoDB → {result.inserted_id}")
        return str(result.inserted_id)

    # ── Vector search: MongoDB Atlas $vectorSearch ─────────────────────────
    def vector_search(self, embedding: list[float], topk: int = 5) -> list[dict]:
        """Run Atlas Vector Search and return top-k metadata records."""
        results = collection.aggregate([
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": embedding,
                    "numCandidates": topk * 10,   # 10x for better recall
                    "limit": topk,
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "name": 1,
                    "company_id": 1,
                    "image_url": 1,
                    "created_at": 1,
                    "score": {"$meta": "vectorSearchScore"},  # similarity score
                }
            }
        ])
        return list(results)

    def total_faces(self) -> int:
        """Return total number of stored face records."""
        return collection.count_documents({})


# ── Pydantic models ────────────────────────────────────────────────────────

class Result(BaseModel):
    name: str
    company_id: int
    rank: int
    distance: float
    created_at: str
    image_url: str


class FaceDetection:
    def __init__(self, det_size=(640, 640)):
        self.app = FaceAnalysis()
        self.app.prepare(ctx_id=0, det_size=det_size)

    def detect_face(self, image):
        return self.app.get(image)

    def load_image(self, image_path: str):
        return np.array(Image.open(image_path).convert("RGB"))


class MetaData(BaseModel):
    name: str
    company_id: int
    image_path: str
    image : Image.Image = None

    model_config = {
        "arbitrary_types_allowed": True
    }


# ── DBMS ───────────────────────────────────────────────────────────────────

class DBMS:
    def __init__(self):
        self.detection = FaceDetection()
        self.metadata_db = MetaDataDB()

    def upload_image(self, metadata: MetaData):

        if metadata.image is not None:
            # Save uploaded image to a temporary path for processing
            temp_path = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            metadata.image.save(temp_path)
            metadata.image_path = temp_path
            logger.info(f"Temporary image saved for processing → {temp_path}")


        preds = self.detection.detect_face(
            self.detection.load_image(metadata.image_path)
        )

        for i, pred in enumerate(preds):
            embedding = pred.get("embedding")
            if embedding is None:
                logger.warning(f"Prediction #{i} has no embedding; skipping.")
                continue

            # ── 1. Upload image to Supabase ────────────────────────────
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_key = f"{metadata.name}_{metadata.company_id}_{ts}_face{i}.jpg"
            image = Image.open(metadata.image_path).convert("RGB")
            public_url = self.metadata_db.save_image(image_key, image)

            # ── 2. Save metadata + embedding to MongoDB ────────────────
            record = {
                "name":       metadata.name,
                "company_id": metadata.company_id,
                "image_url": public_url,
                "created_at": datetime.now().isoformat(),
                "det_score":  float(pred.get("det_score", 0.0)),
                "bbox":       pred["bbox"].tolist() if pred.get("bbox") is not None else [],
                "embedding":  embedding.tolist(),   # stored for $vectorSearch
            }
            self.metadata_db.save_metadata(record)
            logger.info(f"Face #{i} uploaded successfully.")

    def search_face(self, image_path: str = None, image = None, topk: int = 5) -> list[Result]:
        if image is None:
            preds = self.detection.detect_face(
                np.array(Image.open(image_path))
            )
        else:
            preds = self.detection.detect_face(image)

        if self.metadata_db.total_faces() == 0:
            logger.warning("No faces in DB — nothing to search against.")
            return []

        search_results = []

        for pred in preds:
            embedding = pred.get("embedding")
            if embedding is None:
                continue

            hits = self.metadata_db.vector_search(
                embedding=embedding.tolist(),
                topk=topk,
            )

            for rank, hit in enumerate(hits, start=1):
                search_results.append(
                    Result(
                        name=hit["name"],
                        company_id=hit["company_id"],
                        rank=rank,
                        distance=float(hit["score"]),   # Atlas score: higher = more similar
                        created_at=hit["created_at"],
                        image_url=hit["image_url"],
                    )
                )

        return search_results