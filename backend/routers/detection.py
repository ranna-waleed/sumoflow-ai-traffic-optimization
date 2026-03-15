# backend/routers/detection.py
# Endpoints for YOLO vehicle detection

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.yolo_detect import detect_image, get_model, CLASS_NAMES, CLASS_COLORS

router = APIRouter()

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}

# GET /api/detection/status 
@router.get("/status")
def detection_status():
    """Check if model weights exist and are loadable."""
    try:
        get_model()
        return {
            "status":      "ready",
            "model":       "YOLOv8s",
            "classes":     CLASS_NAMES,
            "colors":      CLASS_COLORS,
            "mAP50":       0.478,
            "fps":         208,
        }
    except FileNotFoundError as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "detail": str(e)}
        )


#  POST /api/detection/detect
@router.post("/detect")
async def detect(
    file: UploadFile = File(...),
    confidence: float = Query(default=0.45, ge=0.1, le=0.95),
):
    """
    Upload an image : get bounding boxes + class counts back.
    Accepts: JPEG, PNG, WebP, BMP
    Returns: detections list with bbox coords (pixels + normalized)
    """
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Use JPEG or PNG."
        )

    image_bytes = await file.read()

    if len(image_bytes) > 10 * 1024 * 1024:  # 10 MB limit
        raise HTTPException(status_code=400, detail="Image too large. Max 10MB.")

    try:
        result = detect_image(image_bytes, conf=confidence)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    return result


#  POST /api/detection/detect-batch 
@router.post("/detect-batch")
async def detect_batch(
    files: list[UploadFile] = File(...),
    confidence: float = Query(default=0.45, ge=0.1, le=0.95),
):
    """Upload multiple frames at once — returns a list of results."""
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Max 10 frames per batch.")

    results = []
    for f in files:
        image_bytes = await f.read()
        try:
            results.append(detect_image(image_bytes, conf=confidence))
        except Exception as e:
            results.append({"error": str(e), "filename": f.filename})

    return {"batch_size": len(results), "results": results}