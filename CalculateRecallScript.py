import os
import requests
import json

# Path to ground truth JSON file and OCR service URL
GROUND_TRUTH_PATH = "benchmark_dataset/ground_truth.json"
OCR_SERVICE_URL = "https://shubhamsaini01--ocr-service-fastapi-modal-app.modal.run/ocr"

# Load ground truth data
def load_ground_truth(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Get OCR results from the service for an image
def get_ocr_results(image_path):
    with open(image_path, 'rb') as image_file:
        files = {'files': (os.path.basename(image_path), image_file, 'image/jpeg')}
        response = requests.post(OCR_SERVICE_URL, files=files)
        
        if response.status_code == 200:
            try:
                json_response = response.json()
                if isinstance(json_response, list) and len(json_response) > 0:
                    return json_response[0].get('ocr_results', [])
            except json.JSONDecodeError:
                print(f"Error: Failed to parse JSON response for {image_path}.")
        else:
            print(f"Error: OCR service failed for {image_path}. Status code: {response.status_code}")
    
    return []

# Calculate recall based on OCR results and ground truth
def calculate_recall(ocr_results, ground_truth_data):
    true_positives = 0
    false_negatives = 0

    detected_texts = [result['text'] for result in ocr_results]

    for gt in ground_truth_data:
        if gt['text'] in detected_texts:
            true_positives += 1
        else:
            false_negatives += 1

    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    return recall, true_positives, false_negatives

# Main function to process images and calculate recall
def process_images(image_folder, ground_truth_path, output_file):
    ground_truth = load_ground_truth(ground_truth_path)
    recall_results = {}
    cumulative_true_positives = 0
    cumulative_false_negatives = 0

    # Process each image in the folder
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)

        # Skip non-image files
        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Skipping non-image file: {image_name}")
            continue

        gt_data = ground_truth.get(f"images/{image_name}", [])

        if not gt_data:
            print(f"No ground truth data for {image_name}")
            continue

        # Get OCR results from the service
        ocr_results = get_ocr_results(image_path)
        
        # Calculate recall for the current image
        recall, true_positives, false_negatives = calculate_recall(ocr_results, gt_data)
        recall_results[image_name] = {
            "recall": recall,
            "true_positives": true_positives,
            "false_negatives": false_negatives
        }
        
        # Update cumulative results
        cumulative_true_positives += true_positives
        cumulative_false_negatives += false_negatives

    # Calculate cumulative recall
    cumulative_recall = cumulative_true_positives / (cumulative_true_positives + cumulative_false_negatives) if (cumulative_true_positives + cumulative_false_negatives) > 0 else 0
    recall_results["cumulative"] = {
        "recall": cumulative_recall,
        "true_positives": cumulative_true_positives,
        "false_negatives": cumulative_false_negatives
    }

    # Save the recall results to a file
    with open(output_file, 'w') as f:
        json.dump(recall_results, f, indent=4)
    print(f"Recall results saved to {output_file}")

if __name__ == "__main__":
    # Input paths
    IMAGE_FOLDER = "benchmark_dataset/images"
    GROUND_TRUTH_PATH = "benchmark_dataset/ground_truth.json"
    OUTPUT_FILE = "recall_results.json"

    # Run the recall calculation
    process_images(IMAGE_FOLDER, GROUND_TRUTH_PATH, OUTPUT_FILE)
