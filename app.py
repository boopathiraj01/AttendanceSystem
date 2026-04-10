from fastapi import FastAPI,HTTPException,File,UploadFile,Form,status
from pydantic import BaseModel
from attendance import MetaData, MetaDataDB,DBMS,Result
from PIL import Image
import numpy as np
import io

from  database import status_router
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(status_router)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Attendance System API"}

@app.get("/health/")
def read_health():
    return {"status": "healthy"}

@app.post("/upload/")
async def upload_image(
    name: str = Form(...),
    company_id: int = Form(...),
    file: UploadFile = File(...)
):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        metadata = MetaData(name=name, company_id=company_id, image=image)
        dbms = DBMS()
        dbms.upload_image(metadata)
        return {"message": "Image uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/search/")
async def search_face(frame: UploadFile = File(...)):
    try:
        image_data = await frame.read()
        image = np.array(Image.open(io.BytesIO(image_data)).convert("RGB"))
        dbms = DBMS()
        results = dbms.search_face(image=image)
        return {"results": [result.dict() for result in results]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))