import os
import re
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
import time
from typing import List, Tuple, Dict, Any

# --- Hardcoded Azure Credentials ---
AZURE_ENDPOINT = "https://ocrmak2.cognitiveservices.azure.com/" # Replace with your Azure Endpoint
AZURE_KEY = "9cmzEyZJ1bWwYE2mTdPT4rnneahpbBPqAYFIRgurQQ3uwLeF26JJJQQJ99BDAC5T7U2XJ3w3AAAFACOG18na" # Replace with your Azure Key

# --- Debugging ---
print(f"Azure Endpoint (hardcoded): {AZURE_ENDPOINT}")
print(f"Azure Key (hardcoded): '{AZURE_KEY[:5]}...' if AZURE_KEY else None")
# --- Debugging End ---

# Check if credentials are set (should be, as they are hardcoded)
if not AZURE_KEY or not AZURE_ENDPOINT:
     raise ValueError(
         "Azure Computer Vision credentials are not set. "
         "Ensure AZURE_ENDPOINT and AZURE_KEY are correctly hardcoded."
     )

# Setup Azure Vision client
try:
    computervision_client = ComputerVisionClient(
        AZURE_ENDPOINT,
        CognitiveServicesCredentials(AZURE_KEY)
    )
    print("Azure Computer Vision client initialized successfully.")
except Exception as e:
    print(f"Error initializing Azure Computer Vision client: {e}")
    # Re-raise the exception after printing
    raise

def preprocess_image(image_path: str) -> BytesIO:
    """Preprocess image to enhance OCR performance, including denoising and adaptive thresholding."""
    print(f"Preprocessing image: {image_path}")
    try:
        # Read the image using OpenCV
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            else:
                raise IOError(f"Image file found but could not be read by OpenCV (possibly unsupported format or corrupted): {image_path}")

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # --- Preprocessing Steps ---
        # 1. Noise Reduction: Apply Non-local Means Denoising
        # This is more effective than simple blurring for preserving edges.
        # Adjust h, hForColorComponents, templateWindowSize, searchWindowSize based on image noise.
        # Larger values remove more noise but can smooth out details.
        # For noisy, textured surfaces, experiment with these parameters.
        denoised = cv2.fastNlMeansDenoising(gray, None, h=30, templateWindowSize=7, searchWindowSize=21)
        print("Applied Non-local Means Denoising.")

        # 2. Adaptive Thresholding
        # Applies thresholding based on local pixel neighborhoods.
        # block_size: Size of a pixel neighborhood that is used to calculate the threshold value. Odd numbers only.
        # C: Constant subtracted from the mean or weighted mean. Normally, it is positive.
        block_size = 31 # Increased block size might help with larger variations
        C_value = 10 # Experiment with this value
        thresh = cv2.adaptiveThreshold(
             denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
             cv2.THRESH_BINARY, block_size, C_value
        )
        print(f"Applied Adaptive Thresholding (block_size={block_size}, C={C_value}).")

        # 3. Resize small images for potential better OCR (optional, Azure is often good with scale)
        # Keeping this from your original code as it might help in some cases.
        min_dim = min(thresh.shape[:2])
        if min_dim < 800:
             scale_factor = 800 / min_dim
             # Only resize if scaling up
             if scale_factor > 1.0:
                 width = int(thresh.shape[1] * scale_factor)
                 height = int(thresh.shape[0] * scale_factor)
                 thresh = cv2.resize(thresh, (width, height), interpolation=cv2.INTER_CUBIC)
                 print(f"Resized image to {thresh.shape[1]}x{thresh.shape[0]} for better OCR.")

        # --- End Preprocessing Steps ---

        # Convert processed image (thresholded) to JPEG buffer
        is_success, buf = cv2.imencode(".jpg", thresh)
        if not is_success:
            raise IOError("Failed to encode processed image to JPEG buffer.")

        print("Preprocessing complete.")
        return BytesIO(buf.tobytes())

    except Exception as e:
        print(f"Error during image preprocessing: {e}")
        raise

def extract_numbers_from_text(lines: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """Filter for numeric strings (IDs, codes, measurements) from OCR lines."""
    # Regex to match sequences of one or more digits (\d+)
    # using word boundaries (\b) to match whole numbers.
    # You might need to adjust this regex based on the exact format of your numbers/codes.
    # E.g., if they can include hyphens, letters, etc.
    number_pattern = re.compile(r'\b\d+\b') # Matches whole numbers like 123, 45, 0

    filtered = []
    for text, conf in lines:
        # Find all matches within the text line
        matches = number_pattern.findall(text)
        # Add each found number along with the line's confidence
        for match in matches:
             filtered.append((match, conf)) # Using line confidence for the extracted number

    return filtered


def ocr_image(image_buffer: BytesIO) -> Tuple[str, List[Tuple[str, float]]]:
    """Call Azure OCR Read API and parse results focusing on numeric data."""
    print("Calling Azure OCR Read API...")
    status = "unknown"
    lines_conf = []

    try:
        image_buffer.seek(0)

        # Use the read_in_stream API for bytes input
        read_response = computervision_client.read_in_stream(image_buffer, raw=True)

        # Get the operation location (URL with operation ID)
        operation_location = read_response.headers["Operation-Location"]
        operation_id = operation_location.split("/")[-1]
        print(f"OCR operation ID: {operation_id}")

        # --- Wait for the OCR operation to complete ---
        max_wait_time = 120 # Maximum wait time in seconds
        poll_interval = 1 # Time in seconds to wait between polling
        start_time = time.time()

        while True:
            read_result = computervision_client.get_read_result(operation_id)
            status = read_result.status
            print(f"  OCR Status: {status}")

            if status not in ['notStarted', 'running']:
                break

            if time.time() - start_time > max_wait_time:
                 print(f"Warning: OCR operation timed out after {max_wait_time} seconds.")
                 status = "timedOut"
                 break

            time.sleep(poll_interval)

        # --- Process the results ---
        if status == 'succeeded':
            print("OCR succeeded.")
            if read_result.analyze_result and read_result.analyze_result.read_results:
                for page in read_result.analyze_result.read_results:
                    if page.lines:
                        for line in page.lines:
                            # Use line.text and line.confidence
                            conf = line.confidence if hasattr(line, 'confidence') and line.confidence is not None else 1.0
                            lines_conf.append((line.text.strip(), conf))
            else:
                print("OCR succeeded but no text was detected.")
                status = "succeeded_no_text" # Custom status for logging
        elif status == 'failed':
             print(f"OCR failed with status: {status}")
             if read_result.analyze_result and hasattr(read_result.analyze_result, 'errors'):
                 print("OCR Errors:")
                 for error in read_result.analyze_result.errors:
                     print(f"   Code: {error.code}, Message: {error.message}")
             # Optionally raise an exception here if failure is critical
             # raise Exception(f"Azure OCR operation failed with status: {status}")

        print("OCR complete.")

    except Exception as e:
        print(f"Error during OCR call or processing results: {e}")
        status = f"error: {type(e).__name__}" # Log the type of error
        # Optionally re-raise if needed
        # raise

    return status, lines_conf

def process_image(image_path: str) -> Dict[str, Any]:
    """
    Main function to process a single image file.
    Returns a dictionary containing processing results.
    """
    print(f"\n--- Processing: {image_path} ---")
    results: Dict[str, Any] = {
        "file_path": image_path,
        "status": "not_processed",
        "preprocessing_status": "failed",
        "ocr_status": "failed",
        "all_ocr_lines": [],
        "detected_numbers": [],
        "error": None
    }

    try:
        # Step 1: Preprocess the image
        image_buffer = preprocess_image(image_path)
        results["preprocessing_status"] = "succeeded"

        # Step 2: Perform OCR on the processed image buffer
        ocr_status, ocr_lines = ocr_image(image_buffer)
        results["ocr_status"] = ocr_status
        results["all_ocr_lines"] = ocr_lines # Store all detected lines for context

        # Step 3: Filter for detected numbers if OCR succeeded
        if ocr_status.startswith("succeeded"): # Check for succeeded status (including succeeded_no_text)
             print("Filtering for numbers...")
             number_lines = extract_numbers_from_text(ocr_lines)

             # Sort by confidence (highest first)
             results["detected_numbers"] = sorted(number_lines, key=lambda x: -x[1])
             results["status"] = "completed" if results["detected_numbers"] else "completed_no_numbers_found"

             print("\n--- Detected Numbers (with confidence) ---")
             if not results["detected_numbers"]:
                  print(" No numbers (with at least 1 digit) detected that match the pattern.")
             else:
                  for text, conf in results["detected_numbers"]:
                      print(f"   Text: {text}, Confidence: {conf:.2f}")
        else:
             results["status"] = f"ocr_failed: {ocr_status}"


    except FileNotFoundError as e:
        results["error"] = f"File not found: {e}"
        results["status"] = "failed"
        print(f"Error: {e}")
    except ValueError as e:
        results["error"] = f"Configuration Error: {e}"
        results["status"] = "failed"
        print(f"Configuration Error: {e}")
    except Exception as e:
        results["error"] = f"An unexpected error occurred: {e}"
        results["status"] = "failed"
        print(f"An unexpected error occurred during processing: {e}")

    return results


if __name__ == "__main__":
    # --- Main execution block to process a folder and log results ---
    image_folder = "images" # The name of the folder containing your images
    log_file_name = "ocr_results.txt" # The name of the output log file

    # List of common image file extensions (case-insensitive)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

    print(f"Processing all images in the folder: {image_folder}")
    print(f"Logging results to: {log_file_name}")

    # Ensure the images folder exists
    if not os.path.isdir(image_folder):
        print(f"Error: Folder not found at {image_folder}")
        print("Please ensure the 'images' folder exists in the same directory as the script.")
        exit() # Exit if the folder doesn't exist

    # Open the log file for writing
    try:
        with open(log_file_name, "w", encoding="utf-8") as log_file:
            log_file.write(f"OCR Results Log - Started: {time.ctime()}\n")
            log_file.write("-" * 60 + "\n")

            # Process each file in the folder
            for file_name in os.listdir(image_folder):
                full_file_path = os.path.join(image_folder, file_name)

                # Check if the path is a file and has a valid image extension
                if os.path.isfile(full_file_path):
                    file_extension = os.path.splitext(file_name)[1].lower() # Get extension and convert to lowercase

                    if file_extension in image_extensions:
                        # Process the image file and get results
                        processing_results = process_image(full_file_path)

                        # Log the results
                        log_file.write(f"File: {processing_results['file_path']}\n")
                        log_file.write(f"Overall Status: {processing_results['status']}\n")
                        log_file.write(f"Preprocessing Status: {processing_results['preprocessing_status']}\n")
                        log_file.write(f"OCR Status: {processing_results['ocr_status']}\n")

                        if processing_results["error"]:
                            log_file.write(f"Error Details: {processing_results['error']}\n")

                        log_file.write("Detected Numbers:\n")
                        if processing_results["detected_numbers"]:
                            for text, conf in processing_results["detected_numbers"]:
                                log_file.write(f"  Text: {text}, Confidence: {conf:.2f}\n")
                        else:
                            log_file.write("  None found.\n")

                        log_file.write("All OCR Lines (for reference):\n")
                        if processing_results["all_ocr_lines"]:
                             # Log all lines if needed for debugging, otherwise skip
                             # For brevity in the log, you might skip logging all lines
                             # if processing_results["status"] != "failed":
                             #    for text, conf in processing_results["all_ocr_lines"]:
                             #        log_file.write(f"  Line: '{text}', Confidence: {conf:.2f}\n")
                             # else:
                             #    log_file.write("  (Full OCR lines not available due to failure)\n")
                             pass # Skipping logging all lines by default to keep log cleaner
                        else:
                             log_file.write("  No OCR lines detected.\n")


                        log_file.write("-" * 30 + "\n") # Separator for clarity between files

                    else:
                        log_file.write(f"File: {full_file_path}\n")
                        log_file.write(f"Overall Status: Skipped (Not a recognized image file type)\n")
                        log_file.write("-" * 30 + "\n")

                else:
                    log_file.write(f"Path: {full_file_path}\n")
                    log_file.write(f"Overall Status: Skipped (Not a file)\n")
                    log_file.write("-" * 30 + "\n")

            log_file.write(f"\nOCR Results Log - Finished: {time.ctime()}\n")
            log_file.write("-" * 60 + "\n")

        print(f"\n--- Finished processing all files. Results logged to {log_file_name} ---")

    except IOError as e:
        print(f"Error writing to log file {log_file_name}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during the main processing loop: {e}")