import modal
from fastapi import FastAPI, File, UploadFile, Body
from fastapi.middleware.cors import CORSMiddleware
import easyocr
import cv2
import json
import time
from typing import List

# Initialize Modal
app = modal.App("ocr_service")

# Define the container image to include necessary dependencies and request a GPU
image = modal.Image.debian_slim().pip_install(
    "fastapi", "uvicorn", "easyocr", "opencv-python-headless", "torch"  # Include torch for GPU support
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

@fastapi_app.post("/ocr")
async def perform_ocr(files: List[UploadFile] = File(..., description="List of files to process"), sample_rate: int = Body(1, embed=True)):
    try:
        results_with_benchmark = []
        
        for file in files:
            # Save uploaded file locally
            file_path = f"/tmp/{file.filename}"
            
            # Check file size before saving
            content = await file.read()
            if len(content) > MAX_FILE_SIZE:
                return {"error": f"File size of {file.filename} exceeds maximum limit of 100MB."}
            
            with open(file_path, "wb") as f:
                f.write(content)
            
            # Start benchmark timer
            start_time = time.time()

            # Check if it's a video or an image
            if file.filename.endswith(('.mp4', '.avi')):
                result, frames_processed = process_video(file_path, sample_rate)
            else:
                result = process_image(file_path)
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
        
        return json.dumps(results_with_benchmark)
    
    except Exception as e:
        print(f"Error during OCR processing: {e}")
        return {"error": "An internal server error occurred. Please check the logs for details."}

def process_image(image_path):
    image = cv2.imread(image_path)
    results = reader.readtext(image)
    return format_results(results)

def process_video(video_path, sample_rate):
    results = []
    cap = cv2.VideoCapture(video_path)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0
    frames_processed = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % (frame_rate * sample_rate) == 0:
            frame_results = reader.readtext(frame)
            results.extend(format_results(frame_results))
            frames_processed += 1
        
        frame_count += 1

    cap.release()
    return results, frames_processed

def format_results(results):
    output = []
    for bbox, text, _ in results:
        bbox = [[int(coord[0]), int(coord[1])] for coord in bbox]
        output.append({
            "bounding_box": bbox,
            "text": text
        })
    return output

# Use modal.asgi_app to deploy FastAPI app with Modal and request a GPU
@app.function(image=image, gpu="A10G")  # Request a GPU instance, e.g., "A10G"
@modal.asgi_app()
def fastapi_modal_app():
    return fastapi_app

