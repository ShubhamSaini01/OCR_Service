import os
import time
import requests
import json

# Set the URL of your deployed OCR service
OCR_SERVICE_URL = "https://shubhamsaini01--ocr-service-fastapi-modal-app.modal.run/ocr"

# Path to the downloaded OCR dataset
DATASET_FOLDER = "benchmark_dataset/images"

# Function to perform OCR on an image using the deployed service
def perform_ocr(image_path):
    with open(image_path, 'rb') as file:
        files = {'file': file}
        response = requests.post(OCR_SERVICE_URL, files=files)
        return response

# Function to run the benchmark
def run_benchmark():
    results = []
    total_time = 0

    # Get list of all image files in the dataset folder
    image_files = [f for f in os.listdir(DATASET_FOLDER) if f.endswith(('.jpg', '.png', '.jpeg'))]

    for image_file in image_files:
        image_path = os.path.join(DATASET_FOLDER, image_file)
        print(f"Processing: {image_file}")

        # Measure inference time
        start_time = time.time()
        response = perform_ocr(image_path)
        end_time = time.time()
        
        # Parse the response
        try:
            ocr_result = response.json()
            processing_time = end_time - start_time
            total_time += processing_time

            results.append({
                'image': image_file,
                'processing_time_seconds': processing_time,
                'ocr_result': ocr_result
            })

            print(f"Finished: {image_file} | Time: {processing_time:.2f} seconds")
        except json.JSONDecodeError:
            print(f"Failed to process {image_file}. Response: {response.text}")

    # Calculate average processing time
    average_time = total_time / len(image_files) if image_files else 0

    # Save the results to a JSON file
    with open('ocr_benchmark_results.json', 'w') as outfile:
        json.dump(results, outfile, indent=4)

    print(f"Benchmark completed. Average processing time: {average_time:.2f} seconds per image.")

# Run the benchmark
if __name__ == "__main__":
    run_benchmark()
