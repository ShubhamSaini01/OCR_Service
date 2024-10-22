import os
import requests
import json
import numpy as np
from urllib.parse import urlparse
# Path to ground truth JSON file and OCR service URL
GROUND_TRUTH_PATH = "ground_truth_converted.json"
OCR_SERVICE_URL = "https://shubhamsaini01--keras-ocr-service-fastapi-modal-app.modal.run/ocr"

# Function to extract service name from URL
def get_service_name(url):
    parsed_url = urlparse(url)
    # Extract the service name from the hostname (e.g., 'shubhamsaini01--ocr-service-fastapi-modal-app')
    service_name = parsed_url.hostname.split('--')[1]
    return service_name


# Load ground truth data
def load_ground_truth(file_path):
    print(f"Loading ground truth data from {file_path}...")
    with open(file_path, 'r') as f:
        return json.load(f).get("annotations", {})

# Filter OCR results to remove non-numeric inferences
def filter_numeric_ocr_results(ocr_results):
    print("Filtering OCR results to keep only numeric inferences...")
    numeric_results = []
    for result in ocr_results:
        text = result['text']
        if text.isdigit():  # Check if the text contains only numeric characters
            numeric_results.append(result)
    print(f"Filtered {len(ocr_results) - len(numeric_results)} non-numeric inferences out of {len(ocr_results)} results.")
    return numeric_results

# Get OCR results from the service for an image
def get_ocr_results(image_path):
    print(f"Requesting OCR results for {image_path}...")
    with open(image_path, 'rb') as image_file:
        files = {'files': (os.path.basename(image_path), image_file, 'image/jpeg')}
        response = requests.post(OCR_SERVICE_URL, files=files)
        
        if response.status_code == 200:
            try:
                json_response = response.json()
                if isinstance(json_response, list) and len(json_response) > 0:
                    print(f"Received OCR results for {image_path}: {json_response}")
                    ocr_results = json_response[0].get('ocr_results', [])
                    return filter_numeric_ocr_results(ocr_results)  # Apply numeric filter here
            except json.JSONDecodeError:
                print(f"Error: Failed to parse JSON response for {image_path}.")
        else:
            print(f"Error: OCR service failed for {image_path}. Status code: {response.status_code}")
    
    return []

# IoU calculation between two bounding boxes
def iou(boxA, boxB):
    xA = max(boxA[0][0], boxB[0][0])
    yA = max(boxA[0][1], boxB[0][1])
    xB = min(boxA[2][0], boxB[2][0])
    yB = min(boxA[2][1], boxB[2][1])

    interArea = max(0, xB - xA + 1) * max(0, yA - yA + 1)
    boxAArea = (boxA[2][0] - boxA[0][0] + 1) * (boxA[2][1] - boxA[0][1] + 1)
    boxBArea = (boxB[2][0] - boxB[0][0] + 1) * (boxB[2][1] - boxB[0][1] + 1)

    return interArea / float(boxAArea + boxBArea - interArea)

# Calculate precision and recall based on OCR results and ground truth
def calculate_metrics(ocr_results, ground_truth_data):
    true_positives = 0
    false_negatives = 0

    detected_texts = [result['text'] for result in ocr_results]
    ground_truth_texts = [gt['attributes']['text'] for gt in ground_truth_data]

    print(f"Detected OCR texts: {detected_texts}")
    print(f"Ground truth texts: {ground_truth_texts}")

    for gt in ground_truth_texts:
        if gt in detected_texts:
            true_positives += 1
        else:
            false_negatives += 1

    precision = true_positives / len(detected_texts) if detected_texts else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    print(f"Results: True Positives={true_positives}, False Negatives={false_negatives}")
    
    return precision, recall, true_positives, false_negatives

# Calculate mean Average Precision (mAP)
def calculate_map(precisions):
    return np.mean(precisions) if precisions else 0

# Main function to process images and calculate precision, recall, and mAP
def process_images(image_folder, ground_truth_path, output_file_base):
    print(f"Starting image processing in folder: {image_folder}")
    ground_truth = load_ground_truth(ground_truth_path)
    results = {}
    cumulative_true_positives = 0
    cumulative_false_negatives = 0
    all_precisions = []

    # Process each image in the folder
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)

        # Skip non-image files
        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Skipping non-image file: {image_name}")
            continue

        print(f"Processing {image_name}...")
        gt_data = ground_truth.get(f"images/{image_name}", [])

        if not gt_data:
            print(f"No ground truth data for {image_name}")
            continue

        # Get OCR results from the service
        ocr_results = get_ocr_results(image_path)
        
        if not ocr_results:
            print(f"No OCR results found for {image_name}")
            continue
        
        # Calculate precision, recall, true positives, and false negatives (without false positives for now)
        precision, recall, true_positives, false_negatives = calculate_metrics(ocr_results, gt_data)
        
        print(f"Results for {image_name}: Precision={precision:.4f}, Recall={recall:.4f}")
        
        # Collect results for this image
        results[image_name] = {
            "precision": precision,
            "recall": recall,
            "true_positives": true_positives,
            "false_negatives": false_negatives
        }

        # Update cumulative results
        cumulative_true_positives += true_positives
        cumulative_false_negatives += false_negatives
        all_precisions.append(precision)

    # Calculate cumulative precision, recall, and mAP
    cumulative_precision = cumulative_true_positives / len(all_precisions) if all_precisions else 0
    cumulative_recall = cumulative_true_positives / (cumulative_true_positives + cumulative_false_negatives) if (cumulative_true_positives + cumulative_false_negatives) > 0 else 0
    mAP = calculate_map(all_precisions)

    print(f"Cumulative Precision: {cumulative_precision:.4f}, Cumulative Recall: {cumulative_recall:.4f}, mAP: {mAP:.4f}")
    
    results["cumulative"] = {
        "precision": cumulative_precision,
        "recall": cumulative_recall,
        "mAP": mAP,
        "true_positives": cumulative_true_positives,
        "false_negatives": cumulative_false_negatives
    }

    # Dynamically name the output file
    output_file = f"{output_file_base}_{get_service_name(OCR_SERVICE_URL)}.json"

    # Save the results to a file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    # Input paths
    IMAGE_FOLDER = "benchmark_dataset/images"
    GROUND_TRUTH_PATH = "ground_truth_converted.json"
    OUTPUT_FILE_BASE = "precision_recall_map_results"
    OUTPUT_FILE = "precision_recall_map_results.json"

    # Run the precision, recall, and mAP calculation
    process_images(IMAGE_FOLDER, GROUND_TRUTH_PATH, OUTPUT_FILE_BASE)
