import os
import re
import cv2
import numpy as np
from PIL import Image # Although not used directly for OCR, good to have for potential image ops
# Removed BytesIO as we pass numpy arrays now
import time
from typing import List, Tuple, Dict, Any
import pytesseract # Added for Pytesseract OCR
import pandas as pd # Useful for parsing image_to_data output

# --- Pytesseract Configuration ---
# IMPORTANT: Set the path to your Tesseract executable.
# Find the path after installing Tesseract OCR engine.
# Examples:
# Windows: r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Linux:   '/usr/bin/tesseract' (often found automatically if in PATH)
# macOS:   '/usr/local/bin/tesseract' (if installed via Homebrew)
try:
    # Set the command path (MODIFY THIS PATH AS NEEDED)
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # <--- CHANGE THIS IF NEEDED
    # You can try getting the version to verify the path
    print(f"Pytesseract configured with command: {pytesseract.pytesseract.tesseract_cmd}")
    # tesseract_version = pytesseract.get_tesseract_version() # Uncomment to test
    # print(f"Tesseract version: {tesseract_version}") # Uncomment to test
except pytesseract.TesseractNotFoundError:
    print("\n" + "="*60)
    print(" Tesseract is not installed or the path is incorrect.")
    print(f" Attempted path: '{pytesseract.pytesseract.tesseract_cmd}'")
    print(" Please install Tesseract OCR engine and/or update the path")
    print(" `pytesseract.pytesseract.tesseract_cmd` in the script.")
    print(" Installation guides:")
    print("   Windows: https://github.com/UB-Mannheim/tesseract/wiki")
    print("   macOS: brew install tesseract")
    print("   Linux: sudo apt install tesseract-ocr / sudo dnf install tesseract")
    print("="*60 + "\n")
    raise # Stop execution if Tesseract is not configured
except Exception as e:
    print(f"An unexpected error occurred during Pytesseract configuration: {e}")
    raise
# --- End Pytesseract Configuration ---


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Preprocess image using OpenCV to enhance OCR performance.
    Returns the processed image as a NumPy array (grayscale, thresholded).
    """
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
        denoised = cv2.fastNlMeansDenoising(gray, None, h=30, templateWindowSize=7, searchWindowSize=21)
        print("Applied Non-local Means Denoising.")

        # 2. Adaptive Thresholding
        block_size = 31
        C_value = 10
        thresh = cv2.adaptiveThreshold(
             denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
             cv2.THRESH_BINARY, block_size, C_value
        )
        print(f"Applied Adaptive Thresholding (block_size={block_size}, C={C_value}).")

        # 3. Resize small images (optional, but can help Tesseract)
        min_dim = min(thresh.shape[:2])
        if min_dim < 800:
             scale_factor = 800 / min_dim
             if scale_factor > 1.0:
                 width = int(thresh.shape[1] * scale_factor)
                 height = int(thresh.shape[0] * scale_factor)
                 thresh = cv2.resize(thresh, (width, height), interpolation=cv2.INTER_CUBIC)
                 print(f"Resized image to {thresh.shape[1]}x{thresh.shape[0]} for better OCR.")

        # --- End Preprocessing Steps ---

        # Removed encoding to BytesIO, return the NumPy array directly
        print("Preprocessing complete.")
        return thresh # Return the processed OpenCV image (NumPy array)

    except Exception as e:
        print(f"Error during image preprocessing: {e}")
        raise

def extract_numbers_from_text(lines: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """Filter for numeric strings (IDs, codes, measurements) from OCR lines."""
    # Regex to match sequences of one or more digits (\d+)
    # using word boundaries (\b) to match whole numbers.
    number_pattern = re.compile(r'\b\d+\b') # Matches whole numbers like 123, 45, 0

    filtered = []
    for text, conf in lines:
        # Find all matches within the text line
        matches = number_pattern.findall(text)
        # Add each found number along with the line's confidence
        for match in matches:
             filtered.append((match, conf)) # Using line confidence for the extracted number

    return filtered


def ocr_image(processed_image: np.ndarray) -> Tuple[str, List[Tuple[str, float]]]:
    """
    Perform OCR using Pytesseract on the preprocessed image (NumPy array).
    Returns status and a list of (line_text, average_confidence).
    """
    print("Performing OCR using Pytesseract...")
    status = "unknown"
    lines_conf = []

    try:
        # Use image_to_data to get detailed information including confidence and line numbers
        # Specify language if needed, e.g., lang='eng' (default)
        # Output type DICT makes parsing easier
        ocr_data = pytesseract.image_to_data(processed_image, output_type=pytesseract.Output.DICT, lang='eng')

        n_boxes = len(ocr_data['level']) # Number of detected elements (words, lines, etc.)
        lines = {} # Dictionary to store words per line: {line_num: [(word, conf), ...]}

        if n_boxes == 0:
            print("Pytesseract detected no text boxes.")
            status = "succeeded_no_text"
            return status, lines_conf

        for i in range(n_boxes):
            # Tesseract confidence is 0-100. -1 means not recognized reliably.
            conf = int(ocr_data['conf'][i])
            text = ocr_data['text'][i].strip()

            # Only process words with valid text and positive confidence
            if conf > -1 and text:
                # Get line number associated with this word
                line_num = (ocr_data['page_num'][i], ocr_data['block_num'][i], ocr_data['par_num'][i], ocr_data['line_num'][i])

                if line_num not in lines:
                    lines[line_num] = []
                lines[line_num].append({'text': text, 'conf': conf})

        # Reconstruct lines and calculate average confidence
        for line_num in sorted(lines.keys()):
            words_in_line = lines[line_num]
            line_text = " ".join([word['text'] for word in words_in_line])

            # Calculate average confidence for the line, ignoring -1 confidences (which we filtered)
            # Normalize Tesseract's 0-100 confidence to 0.0-1.0
            valid_confs = [word['conf'] for word in words_in_line if word['conf'] > -1] # Should always be > -1 here
            if valid_confs:
                avg_conf = (sum(valid_confs) / len(valid_confs)) / 100.0
            else:
                avg_conf = 0.0 # Or handle as needed if no words have confidence

            lines_conf.append((line_text, avg_conf))

        if lines_conf:
            print("Pytesseract OCR succeeded.")
            status = "succeeded"
        else:
             print("Pytesseract OCR completed but no valid text lines reconstructed.")
             status = "succeeded_no_text" # Or a more specific status

    except pytesseract.TesseractNotFoundError:
        print("Error: Tesseract executable not found or path is incorrect.")
        print(f"Check the configuration: pytesseract.pytesseract.tesseract_cmd = '{pytesseract.pytesseract.tesseract_cmd}'")
        status = "error_tesseract_not_found"
    except pytesseract.TesseractError as te:
        print(f"Error during Pytesseract OCR execution: {te}")
        status = f"error_tesseract: {type(te).__name__}"
    except Exception as e:
        print(f"An unexpected error occurred during Pytesseract OCR: {e}")
        status = f"error: {type(e).__name__}"

    print("OCR complete.")
    return status, lines_conf


def process_image(image_path: str) -> Dict[str, Any]:
    """
    Main function to process a single image file using Pytesseract.
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
        # Step 1: Preprocess the image (returns NumPy array)
        processed_np_image = preprocess_image(image_path)
        results["preprocessing_status"] = "succeeded"

        # Step 2: Perform OCR on the processed NumPy image
        ocr_status, ocr_lines = ocr_image(processed_np_image)
        results["ocr_status"] = ocr_status
        results["all_ocr_lines"] = ocr_lines # Store all detected lines for context

        # Step 3: Filter for detected numbers if OCR succeeded
        if ocr_status.startswith("succeeded"): # Check for succeeded status (including succeeded_no_text)
             print("Filtering for numbers...")
             number_lines = extract_numbers_from_text(ocr_lines)

             # Sort by confidence (highest first)
             # Note: Confidence is now the *average line confidence* from Pytesseract (0.0-1.0)
             results["detected_numbers"] = sorted(number_lines, key=lambda x: -x[1])
             results["status"] = "completed" if results["detected_numbers"] else "completed_no_numbers_found"

             print("\n--- Detected Numbers (with avg line confidence) ---")
             if not results["detected_numbers"]:
                 print(" No numbers (with at least 1 digit) detected that match the pattern.")
             else:
                 for text, conf in results["detected_numbers"]:
                     # Confidence is normalized average line confidence
                     print(f"   Text: {text}, Avg Line Confidence: {conf:.2f}")
        else:
             results["status"] = f"ocr_failed: {ocr_status}"
             # Log specific Tesseract errors if available
             if ocr_status.startswith("error"):
                 results["error"] = f"OCR Error: {ocr_status}"


    except FileNotFoundError as e:
        results["error"] = f"File not found: {e}"
        results["status"] = "failed_file_not_found"
        results["preprocessing_status"] = "skipped"
        print(f"Error: {e}")
    except IOError as e: # Catch errors from reading/processing the image
        results["error"] = f"Image Processing Error: {e}"
        results["status"] = "failed_preprocessing"
        results["preprocessing_status"] = "failed"
        print(f"Error: {e}")
    except pytesseract.TesseractNotFoundError as e: # Catch configuration error early if possible
        results["error"] = f"Configuration Error: {e}"
        results["status"] = "failed_configuration"
        print(f"Configuration Error: {e}")
    except Exception as e:
        results["error"] = f"An unexpected error occurred: {e}"
        results["status"] = "failed_unexpected"
        # Update specific statuses if possible
        if results["preprocessing_status"] != "succeeded":
             results["preprocessing_status"] = "failed"
        if results["ocr_status"] == "unknown":
             results["ocr_status"] = "failed"
        print(f"An unexpected error occurred during processing: {e}")

    return results


if __name__ == "__main__":
    # --- Main execution block to process a folder and log results ---
    image_folder = "images" # The name of the folder containing your images
    log_file_name = "ocr_results_pytesseract.txt" # Changed log file name

    # List of common image file extensions (case-insensitive)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

    print(f"Processing all images in the folder: {image_folder} using Pytesseract")
    print(f"Logging results to: {log_file_name}")

    # Ensure the images folder exists
    if not os.path.isdir(image_folder):
        print(f"Error: Folder not found at {os.path.abspath(image_folder)}")
        print("Please ensure the 'images' folder exists in the same directory as the script.")
        exit() # Exit if the folder doesn't exist

    # Open the log file for writing
    try:
        with open(log_file_name, "w", encoding="utf-8") as log_file:
            log_file.write(f"Pytesseract OCR Results Log - Started: {time.ctime()}\n")
            log_file.write("-" * 60 + "\n")

            processed_count = 0
            skipped_count = 0
            failed_count = 0

            # Process each file in the folder
            all_files = os.listdir(image_folder)
            print(f"Found {len(all_files)} items in the '{image_folder}' directory.")

            for file_name in all_files:
                full_file_path = os.path.join(image_folder, file_name)

                # Check if the path is a file and has a valid image extension
                if os.path.isfile(full_file_path):
                    file_extension = os.path.splitext(file_name)[1].lower() # Get extension and convert to lowercase

                    if file_extension in image_extensions:
                        # Process the image file and get results
                        processing_results = process_image(full_file_path)

                        # Update counters
                        if processing_results["status"].startswith("completed"):
                             processed_count += 1
                        elif processing_results["status"].startswith("failed"):
                             failed_count += 1
                        # Handle other statuses if needed

                        # Log the results
                        log_file.write(f"File: {processing_results['file_path']}\n")
                        log_file.write(f"Overall Status: {processing_results['status']}\n")
                        log_file.write(f"Preprocessing Status: {processing_results['preprocessing_status']}\n")
                        log_file.write(f"OCR Status: {processing_results['ocr_status']}\n")

                        if processing_results["error"]:
                             log_file.write(f"Error Details: {processing_results['error']}\n")

                        log_file.write("Detected Numbers (Avg Line Confidence):\n")
                        if processing_results["detected_numbers"]:
                             for text, conf in processing_results["detected_numbers"]:
                                 log_file.write(f"   Text: {text}, Confidence: {conf:.2f}\n")
                        else:
                             log_file.write("   None found.\n")

                        # Log all lines only if specifically needed or on failure for debugging
                        if processing_results["status"].startswith("failed") or True: # Set to False to hide full lines on success
                            log_file.write("All OCR Lines (for reference):\n")
                            if processing_results["all_ocr_lines"]:
                                for text, conf in processing_results["all_ocr_lines"]:
                                     log_file.write(f"   Line: '{text}', Avg Confidence: {conf:.2f}\n")
                            else:
                                log_file.write("   No OCR lines detected or reconstructed.\n")

                        log_file.write("-" * 30 + "\n") # Separator for clarity between files

                    else:
                        skipped_count += 1
                        print(f"Skipping non-image file: {file_name}")
                        log_file.write(f"File: {full_file_path}\n")
                        log_file.write(f"Overall Status: Skipped (Not a recognized image file type: {file_extension})\n")
                        log_file.write("-" * 30 + "\n")

                else:
                    skipped_count += 1
                    print(f"Skipping non-file item: {file_name}")
                    log_file.write(f"Path: {full_file_path}\n")
                    log_file.write(f"Overall Status: Skipped (Not a file)\n")
                    log_file.write("-" * 30 + "\n")

            # Write summary to log
            log_file.write("\n" + "="*60 + "\n")
            log_file.write("Processing Summary:\n")
            log_file.write(f"  Successfully processed (numbers found or not): {processed_count}\n")
            log_file.write(f"  Failed during processing: {failed_count}\n")
            log_file.write(f"  Skipped (not image or not file): {skipped_count}\n")
            log_file.write(f"  Total items considered: {len(all_files)}\n")
            log_file.write("="*60 + "\n")
            log_file.write(f"\nPytesseract OCR Results Log - Finished: {time.ctime()}\n")
            log_file.write("-" * 60 + "\n")

        print(f"\n--- Finished processing. ---")
        print(f"  Processed: {processed_count}")
        print(f"  Failed:    {failed_count}")
        print(f"  Skipped:   {skipped_count}")
        print(f"--- Results logged to {log_file_name} ---")

    except IOError as e:
        print(f"Error writing to log file {log_file_name}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during the main processing loop: {e}")