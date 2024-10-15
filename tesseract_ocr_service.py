import modal
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pytesseract
import cv2
import os
from typing import List

# Initialize Modal
app = modal.App("tesseract_ocr_service")

# Define the container image with Tesseract and OpenCV
image = modal.Image.debian_slim().apt_install(
    "tesseract-ocr", "libtesseract-dev"
).pip_install(
    "fastapi", "pytesseract", "uvicorn", "opencv-python-headless"
)

# Create FastAPI app
fastapi_app = FastAPI()

# Enable CORS
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Maximum file size for upload
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB

@fastapi_app.post("/ocr")
async def perform_tesseract_ocr(
    files: List[UploadFile] = File(..., description="List of files to process")
):
    try:
        results_with_benchmark = []
        for file in files:
            # Save uploaded file locally
            file_path = f"/tmp/{file.filename}"

            # Check file size before saving
            content = await file.read()
            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(status_code=413, detail=f"File size of {file.filename} exceeds 100MB.")

            with open(file_path, "wb") as f:
                f.write(content)

            # Start OCR with Tesseract
            ocr_text = run_tesseract_ocr(file_path)

            # Append result for each file
            results_with_benchmark.append({
                "file_name": file.filename,
                "ocr_text": ocr_text
            })

        return results_with_benchmark

    except HTTPException as e:
        raise e
    except Exception as e:
        return {"detail": f"Error during OCR processing: {str(e)}"}

def run_tesseract_ocr(image_path: str) -> str:
    """
    Run Tesseract OCR on the given image and return the extracted text.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read the image {image_path}.")
    
    # Perform OCR using Tesseract
    ocr_text = pytesseract.image_to_string(img)
    return ocr_text

# Use modal.asgi_app to deploy FastAPI app with Modal
@app.function(image=image)
@modal.asgi_app()
def fastapi_modal_app():
    return fastapi_app
