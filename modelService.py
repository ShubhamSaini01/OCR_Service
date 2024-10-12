import modal
from fastapi import FastAPI, File, UploadFile, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import easyocr
import cv2
import time
from typing import List

# Global flag to control logging for the current request
LOGGING_ENABLED = False  # Default value, can be overridden per request

# Initialize Modal
app = modal.App("ocr_service")

# Define the container image to include necessary dependencies and request a GPU
image = modal.Image.debian_slim().pip_install(
    "fastapi", "uvicorn", "easyocr", "opencv-python-headless", "torch"
)

# Create the FastAPI app and rename it to avoid conflict
fastapi_app = FastAPI()

# Set up CORS middleware on the FastAPI app
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the EasyOCR reader at the global scope and enable GPU (if available)
reader = easyocr.Reader(['en'], gpu=True)  # Enable GPU

# Set the maximum file size limit (e.g., 100MB)
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB

# Available OCR models
available_models = ["easyocr"]

# Utility function to log messages
def log_message(message: str):
    if LOGGING_ENABLED:
        print(message)

@fastapi_app.post("/ocr")
async def perform_ocr(
    files: List[UploadFile] = File(..., description="List of files to process"),
    sample_rate: int = Body(1, embed=True),
    model_name: str = Body("easyocr", embed=True),
    logging_enabled: bool = Body(False, embed=True)
):
    global LOGGING_ENABLED  # Use global variable to control logging
    LOGGING_ENABLED = logging_enabled  # Set logging status based on the request parameter

    if model_name not in available_models:
        raise HTTPException(status_code=400, detail=f"Unsupported OCR model: {model_name}. Available models are {available_models}.")

    try:
        results_with_benchmark = []

        for file in files:
            # Save uploaded file locally
            file_path = f"/tmp/{file.filename}"
            
            # Check file size before saving
            content = await file.read()
            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(status_code=413, detail=f"File size of {file.filename} exceeds maximum limit of 100MB.")
            
            with open(file_path, "wb") as f:
                f.write(content)

            # Start benchmark timer
            start_time = time.time()

            # Check if it's a video or an image
            if file.filename.endswith(('.mp4', '.avi')):
                result, frames_processed = process_video(file_path, sample_rate, model_name)
            else:
                result = process_image(file_path, model_name)
                frames_processed = None

            # End benchmark timer
            end_time = time.time()
            processing_time = end_time - start_time

            # Append result for each file
            results_with_benchmark.append({
                "file_name": file.filename,
                "processing_time_seconds": processing_time,
                "frames_processed": frames_processed,
                "ocr_results": result
            })
        
        return results_with_benchmark
    
    except HTTPException as e:
        raise e
    except Exception as e:
        log_message(f"Error during OCR processing for {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail="An internal server error occurred. Please check the logs for details.")
    finally:
        LOGGING_ENABLED = False

def process_image(image_path, model_name):
    if model_name == "easyocr":
        results = reader.readtext(image_path)
        return format_results(results)

    else:
        raise ValueError(f"Unsupported OCR model: {model_name}")

def process_video(video_path, sample_rate, model_name):
    # Similar to process_image but processes frames in the video
    pass  # Implement video processing similarly to process_image

def format_results(results):
    output = []
    for bbox, text, _ in results:
        bbox = [[int(coord[0]), int(coord[1])] for coord in bbox]
        output.append({"bounding_box": bbox, "text": text})
    return output

# Use modal.asgi_app to deploy FastAPI app with Modal and request a GPU
@app.function(image=image, gpu="A10G")
@modal.asgi_app()
def fastapi_modal_app():
    return fastapi_app
