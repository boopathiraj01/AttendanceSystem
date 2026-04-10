from v1.AttendanceSystem.AttendanceSystem.attendance import MetaDataDB, client, collection
from dotenv import load_dotenv
from fastapi import APIRouter
from bson import ObjectId

load_dotenv()
status_router = APIRouter()
DB = MetaDataDB()

def serialize_doc(doc) -> dict:
    """Convert MongoDB document to JSON-serializable dict."""
    if doc is None:
        return None
    doc["_id"] = str(doc["_id"])  # Convert ObjectId to string
    return doc

@status_router.get("/status/")
def get_status():
    return {"total faces": DB.total_faces()}

@status_router.get("/faces/")
def list_faces(start: int = 0, limit: int = 10):
    records = collection.find({}).skip(start).limit(limit)
    return {"faces": [serialize_doc(doc) for doc in records]}  # Iterate cursor, serialize each doc

@status_router.get("/face/{company_id}")
def get_face(company_id: str):

    if len(company_id) < 2:
        company_id = int(company_id)
    record = collection.find_one({"company_id": company_id})
    if not record:
        return {"error": "Face not found"}
    return serialize_doc(record)

@status_router.get("/face/{company_id}/{name}")
def get_face_by_name(company_id: str, name: str):
    if len(company_id) < 2:
        company_id = int(company_id)
    record = collection.find_one({"company_id": company_id, "name": name})
    if not record:
        return {"error": "Face not found"}
    return serialize_doc(record)