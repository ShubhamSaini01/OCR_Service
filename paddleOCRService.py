import modal
from fastapi import FastAPI, File, UploadFile, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import time
from typing import List

# Initialize Modal
app = modal.App("paddle_ocr_service")

# Define the container image with dependencies, skipping conflicting packages
image = modal.Image.debian_slim().apt_install(
    "libgl1", 
    "libglib2.0-0", 
    "libgstreamer1.0-0"
).pip_install(
    "fastapi", 
    "uvicorn", 
    "opencv-python-headless", 
    "paddlepaddle",
    "paddleocr>=2.0.1",
    "albumentations"
)

# Create the FastAPI app and set up CORS
fastapi_app = FastAPI()
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy load PaddleOCR within Modal container
def load_paddle_ocr():
    with image.imports():
        from paddleocr import PaddleOCR
        return PaddleOCR(use_angle_cls=True, lang='en')

# Set the maximum file size limit (100MB)
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB

@fastapi_app.post("/ocr")
async def perform_paddleocr(
    files: List[UploadFile] = File(..., description="List of files to process"),
    logging_enabled: bool = Body(False, embed=True)
):
    try:
        # Lazy load PaddleOCR only when this endpoint is called
        paddle_reader = load_paddle_ocr()
        results_with_benchmark = []

        for file in files:
            file_path = f"/tmp/{file.filename}"
            content = await file.read()

            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(status_code=413, detail=f"File size of {file.filename} exceeds maximum limit of 100MB.")
            
            with open(file_path, "wb") as f:
                f.write(content)

            # Start benchmark timer
            start_time = time.time()
            result = process_image_with_paddleocr(paddle_reader, file_path)
            # End benchmark timer
            end_time = time.time()
            processing_time = end_time - start_time

            results_with_benchmark.append({
                "file_name": file.filename,
                "processing_time_seconds": processing_time,
                "ocr_results": result
            })
        
        return results_with_benchmark
    
    except HTTPException as e:
        raise e
    except Exception as e:
        error_message = f"Error during OCR processing: {str(e)}"
        print(error_message)
        return {"detail": error_message}

# Process image with PaddleOCR
def process_image_with_paddleocr(paddle_reader, image_path):

    results = paddle_reader.ocr(image_path)
    formatted_results = [{"text": line[1][0], "bounding_box": line[0]} for line in results]
    return formatted_results

# Use modal.asgi_app to deploy FastAPI app with Modal and request a GPU
@app.function(image=image, gpu="A10G")
@modal.asgi_app()
def fastapi_modal_app():
    return fastapi_app
