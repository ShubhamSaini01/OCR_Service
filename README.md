# OCR Service using EasyOCR on Modal
## A lightweight OCR (Optical Character Recognition) service deployed on Modal using EasyOCR with GPU (A10G) support to process images and videos.

## API Endpoint
    URL: https://shubhamsaini01--ocr-service-fastapi-modal-app.modal.run/ocr
## Usage Example
### bash
    curl -X POST "https://shubhamsaini01--ocr-service-fastapi-modal-app.modal.run/ocr" -F "file=@/path/to/your/image.jpg"
    
## Benchmarking
     Dataset: Kaggle OCR Dataset
     Average inference time: 3.52 seconds per image
    
## To-Do
    Confirm GPU utilization during benchmarking.
    Research state-of-the-art (SOTA) OCR methods.
    Create a detailed README file.
