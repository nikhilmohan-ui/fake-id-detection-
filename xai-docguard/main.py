import sys
from pathlib import Path

# Ensure backend directory is in path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from modules.preprocess import preprocess_document
from modules.ocr_module import extract_fields
from modules.tamper_detection import analyze_tampering
from modules.face_match import match_faces
from modules.risk_engine import calculate_risk
from modules.xai_explain import generate_explanation
from utils.helpers import load_image_from_bytes

app = FastAPI(
    title="XAI-DocGuard",
    description="Explainable AI Document Fraud Screening System - SIH26188",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok", "service": "XAI-DocGuard", "team": "ForensiX"}

@app.post("/analyze")
async def analyze_document(
    document: UploadFile = File(..., description="Identity document image"),
    selfie: UploadFile = File(..., description="Live selfie of the person")
):
    try:
        doc_bytes = await document.read()
        selfie_bytes = await selfie.read()

        doc_img = load_image_from_bytes(doc_bytes)
        selfie_img = load_image_from_bytes(selfie_bytes)

        if doc_img is None or selfie_img is None:
            raise HTTPException(status_code=400, detail="Invalid image files")

        # 1. Preprocess
        processed = preprocess_document(doc_img)

        # 2. OCR
        ocr_fields, _ = extract_fields(processed)

        # 3. Tamper detection
        tamper_result = analyze_tampering(processed)

        # 4. Face matching
        face_result = match_faces(doc_img, selfie_img)

        # 5. Risk score
        risk_result = calculate_risk(ocr_fields, tamper_result, face_result)

        # 6. Explainability
        explanation = generate_explanation(
            tamper_result.get("heatmap"),
            risk_result,
            face_result
        )

        response = {
            "status": "success",
            "document_type": ocr_fields.get("document_type"),
            "extracted_fields": {
                "name": ocr_fields.get("name"),
                "document_number": ocr_fields.get("document_number"),
                "date_of_birth": ocr_fields.get("date_of_birth"),
                "expiry_date": ocr_fields.get("expiry_date"),
                "nationality": ocr_fields.get("nationality"),
                "gender": ocr_fields.get("gender")
            },
            "tamper_analysis": {
                "score": tamper_result["tamper_score"],
                "reasons": tamper_result["reasons"]
            },
            "face_match": {
                "verified": face_result["verified"],
                "similarity": face_result["similarity"],
                "message": face_result["message"]
            },
            "risk": {
                "score": risk_result["risk_score"],
                "decision": risk_result["decision"],
                "color": risk_result["color"],
                "reasons": risk_result["reasons"]
            },
            "explanation": explanation
        }

        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
