# OCR Service using EasyOCR on Modal
## A lightweight OCR (Optical Character Recognition) service deployed on Modal using EasyOCR with GPU (A10G) support to process images and videos.

## API Endpoint
    URL: https://shubhamsaini01--ocr-service-fastapi-modal-app.modal.run/ocr
## Usage Example
### 1. Single Image Processing
You can send a single image to the OCR service and retrieve the results by making a POST request. Here's how you can do it via curl or using the provided Python script:

Curl Command for Single Image:
### bash
    curl -X POST "https://shubhamsaini01--ocr-service-fastapi-modal-app.modal.run/ocr" \-F "files=@/path/to/your/image.jpg"
Replace /path/to/your/image.jpg with the actual path to the image file you want to process.
## Python Script for Single Image:
You can also use the provided Python function to send a single image:

### python
    def perform_ocr_single(image_path):
        with open(image_path, 'rb') as file:
            files = {'files': file}
            response = requests.post(OCR_SERVICE_URL, files=files)
            return response
### 2. Batch Image Processing 
You can process multiple images in a single request by sending them in a batch. The batch size can be customized as needed.

Curl Command for Batch Processing:
### bash
    curl -X POST "https://shubhamsaini01--ocr-service-fastapi-modal-app.modal.run/ocr" \-F "files=@/path/to/image1.jpg" \-F "files=@/path/to/image2.jpg"
You can send as many images as needed by repeating the -F flag for each image.

## Python Script for Batch Processing:
Use the following Python function to send a batch of images:

### python
    def perform_ocr_batch(image_paths):
        files = [('files', (os.path.basename(path), open(path, 'rb'), 'image/jpeg')) for path in image_paths]
        response = requests.post(OCR_SERVICE_URL, files=files)
        return response
This function sends multiple images in a single request by uploading them as a batch.
    
## Benchmarking
     Dataset: Kaggle OCR Dataset
     Average inference time: 3.52 seconds per image
    
## To-Do
    Confirm GPU utilization during benchmarking.
    Research state-of-the-art (SOTA) OCR methods.
    Create a detailed README file.
