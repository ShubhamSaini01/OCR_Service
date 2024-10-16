import modal
from fastapi import FastAPI, File, UploadFile, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import time
from typing import List

# Initialize Modal app
app = modal.App("keras_ocr_service")

# Define the container image with lazy imports inside the image context
image = modal.Image.debian_slim().apt_install(
    "libgl1", 
    "libglib2.0-0", 
    "libgstreamer1.0-0"
).pip_install(
    "fastapi", 
    "uvicorn",
    "tensorflow==2.15",
    "keras-ocr" 
)

# Create FastAPI app
fastapi_app = FastAPI()

# Set up CORS middleware
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Maximum file size limit (e.g., 100MB)
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB

@fastapi_app.post("/ocr")
async def perform_ocr(
    files: List[UploadFile] = File(..., description="List of files to process"),
    logging_enabled: bool = Body(False, embed=True)
):
    try:
        results_with_benchmark = []
        # Lazy load Keras-OCR and TensorFlow within the Modal image context
        with image.imports():
            import keras_ocr
        
            # Create pipeline inside the imports context
            pipeline = keras_ocr.pipeline.Pipeline()

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

                # Process image using Keras-OCR
                result = process_image_with_keras_ocr(file_path, pipeline)

                # End benchmark timer
                end_time = time.time()
                processing_time = end_time - start_time

                # Append result for each file
                results_with_benchmark.append({
                    "file_name": file.filename,
                    "processing_time_seconds": processing_time,
                    "ocr_results": result
                })
        
        return results_with_benchmark

    except HTTPException as e:
        raise e
    except Exception as e:
        # Check if 'file' exists in the current scope
        if 'file' in locals():
            error_message = f"Error during OCR processing for {file.filename}: {str(e)}"
        else:
            error_message = f"Error during OCR processing: {str(e)}"
        print(error_message)
        return {"detail": error_message}

def process_image_with_keras_ocr(image_path, pipeline):
    # Use Keras-OCR to process the image
    with image.imports():
        import keras_ocr
        
    images = [keras_ocr.tools.read(image_path)]
    prediction_groups = pipeline.recognize(images)
    
    # Format the results
    formatted_results = []
    for predictions in prediction_groups:
        for prediction in predictions:
            text, box = prediction
            formatted_results.append({
                "text": text,
                "bounding_box": box.tolist()
            })
    
    return formatted_results

# Use modal.asgi_app to deploy FastAPI app with Modal and request a GPU
@app.function(image=image, gpu="A10G")  # Adjust GPU as needed
@modal.asgi_app()
def fastapi_modal_app():
    return fastapi_app
