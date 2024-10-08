import os
import requests
import time
import json
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Set the URL of your deployed OCR service
OCR_SERVICE_URL = "https://shubhamsaini01--ocr-service-fastapi-modal-app.modal.run/ocr"

# Path to the folder containing images
DATASET_FOLDER = "benchmark_dataset/images"

# File to store the benchmark results
OUTPUT_FILE = "ocr_benchmark_results.json"

# Create a session with retry logic
session = requests.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504, 429])
session.mount('https://', HTTPAdapter(max_retries=retries))

# Send a batch of images for OCR
def perform_ocr_batch(image_paths):
    files = [('files', (os.path.basename(path), open(path, 'rb'), 'image/jpeg')) for path in image_paths]
    response = session.post(OCR_SERVICE_URL, files=files)
    return response

# Send a single image for OCR
def perform_ocr_single(image_path):
    with open(image_path, 'rb') as file:
        files = {'files': file}
        response = session.post(OCR_SERVICE_URL, files=files)
        return response

# Benchmark individual image processing
def benchmark_individual():
    results = []
    image_files = [os.path.join(DATASET_FOLDER, img) for img in os.listdir(DATASET_FOLDER) if img.endswith(('jpg', 'png'))]
    
    # Start total time tracking for individual processing
    total_start_time = time.time()

    for image_file in image_files:
        print(f"Processing single image: {image_file}")
        start_time = time.time()
        response = perform_ocr_single(image_file)
        end_time = time.time()

        processing_time = end_time - start_time
        result = {
            "image": os.path.basename(image_file),
            "processing_time_seconds": processing_time,
            "ocr_result": response.json()
        }
        results.append(result)
    
    # End total time tracking
    total_end_time = time.time()
    total_processing_time = total_end_time - total_start_time

    print(f"Total time taken for individual processing: {total_processing_time:.2f} seconds")

    return results, total_processing_time

# Benchmark batch processing
def benchmark_batch(batch_size):
    results = []
    image_files = [os.path.join(DATASET_FOLDER, img) for img in os.listdir(DATASET_FOLDER) if img.endswith(('jpg', 'png'))]
    
    # Start total time tracking for batch processing
    total_start_time = time.time()

    for i in range(0, len(image_files), batch_size):
        batch = image_files[i:i + batch_size]
        print(f"Processing batch of {len(batch)} images")
        start_time = time.time()
        response = perform_ocr_batch(batch)
        end_time = time.time()

        processing_time = end_time - start_time
        result = {
            "batch_size": len(batch),
            "processing_time_seconds": processing_time,
            "ocr_result": response.json()
        }
        results.append(result)
    
    # End total time tracking
    total_end_time = time.time()
    total_processing_time = total_end_time - total_start_time

    print(f"Total time taken for batch processing: {total_processing_time:.2f} seconds")

    return results, total_processing_time

# Run both individual and batch benchmarks
def run_benchmark(batch_size):
    benchmark_results = {}

    # Benchmark individual image processing
    print("Starting individual image processing benchmark...")
    individual_results, individual_total_time = benchmark_individual()
    benchmark_results["individual_processing"] = {
        "results": individual_results,
        "total_time_seconds": individual_total_time
    }

    # Benchmark batch image processing
    print(f"Starting batch processing benchmark with batch size {batch_size}...")
    batch_results, batch_total_time = benchmark_batch(batch_size)
    benchmark_results["batch_processing"] = {
        "results": batch_results,
        "total_time_seconds": batch_total_time
    }

    # Save the benchmark results to a JSON file
    with open(OUTPUT_FILE, 'w') as outfile:
        json.dump(benchmark_results, outfile, indent=4)

    print(f"Benchmark completed. Results saved to {OUTPUT_FILE}")

# Run the benchmark with a batch size of 5 (or any size you want to test)
if __name__ == "__main__":
    run_benchmark(batch_size=5)
