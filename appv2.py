import os
import re
import cv2
import numpy as np
# from PIL import Image # PIL not strictly needed if using cv2 for image ops
from io import BytesIO
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
import time
from typing import List, Tuple, Dict, Any

# --- Hardcoded Azure Credentials ---
# WARNING: Hardcoding sensitive credentials is a security risk and NOT recommended for production.
# It is best practice to use environment variables or a secure configuration management system.
# Replace with your actual endpoint and key
AZURE_ENDPOINT = "https://ocrmak2.cognitiveservices.azure.com/" # Replace with your Azure Endpoint
AZURE_KEY = "9cmzEyZJ1bWwYE2mTdPT4rnneahpbBPqAYFIRgurQQ3uwLeF26JJJQQJ99BDAC5T7U2XJ3w3AAAFACOG18na" # Replace with your Azure Key
# --- End Hardcoded Credentials ---

# --- Debugging ---
print(f"Azure Endpoint: {AZURE_ENDPOINT}")
# Mask the key for safety in logs
print(f"Azure Key: '{AZURE_KEY[:5]}...' if AZURE_KEY and len(AZURE_KEY) > 5 else 'Provided but short/empty'")
# --- Debugging End ---

# Check if credentials are set
if not AZURE_KEY or AZURE_KEY == "YOUR_AZURE_KEY" or \
   not AZURE_ENDPOINT or AZURE_ENDPOINT == "YOUR_AZURE_ENDPOINT":
     raise ValueError(
         "Azure Computer Vision credentials are not set or are still placeholders. "
         "Please replace 'YOUR_AZURE_ENDPOINT' and 'YOUR_AZURE_KEY' with your actual credentials."
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

# --- Image Preprocessing Options ---

def preprocess_image_detailed(image_path: str) -> BytesIO:
    """
    Preprocess image with denoising and adaptive thresholding.
    NOTE: This might be too aggressive for textured surfaces like metal.
    """
    print(f"Preprocessing image (detailed): {image_path}")
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

        # --- Optional: Save grayscale for comparison ---
        # cv2.imwrite(f"{os.path.splitext(image_path)[0]}_gray.png", gray)

        # 1. Noise Reduction: Non-local Means Denoising (Experiment with 'h')
        # Lower 'h' values (e.g., 5, 10) are less aggressive.
        h_value = 10 # Reduced from 30, tune this based on results
        denoised = cv2.fastNlMeansDenoising(gray, None, h=h_value, templateWindowSize=7, searchWindowSize=21)
        print(f"Applied Non-local Means Denoising (h={h_value}).")

        # --- Optional: Save denoised for comparison ---
        # cv2.imwrite(f"{os.path.splitext(image_path)[0]}_denoised.png", denoised)

        # 2. Adaptive Thresholding (Experiment with block_size, C)
        block_size = 31 # Must be odd
        C_value = 10
        thresh = cv2.adaptiveThreshold(
             denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
             cv2.THRESH_BINARY, block_size, C_value # Or try cv2.THRESH_BINARY_INV
        )
        print(f"Applied Adaptive Thresholding (block_size={block_size}, C={C_value}).")

        # --- Optional: Save thresholded image to inspect ---
        cv2.imwrite(f"{os.path.splitext(image_path)[0]}_preprocessed_detailed.png", thresh)


        # 3. Optional Resize (Azure often handles scale well)
        min_dim = min(thresh.shape[:2])
        if min_dim < 800:
             scale_factor = 800 / min_dim
             if scale_factor > 1.0:
                 width = int(thresh.shape[1] * scale_factor)
                 height = int(thresh.shape[0] * scale_factor)
                 # Use INTER_CUBIC or INTER_LINEAR for enlarging
                 thresh = cv2.resize(thresh, (width, height), interpolation=cv2.INTER_CUBIC)
                 print(f"Resized image to {thresh.shape[1]}x{thresh.shape[0]} for OCR.")

        # Convert processed image to PNG buffer (lossless)
        is_success, buf = cv2.imencode(".png", thresh)
        if not is_success:
            raise IOError("Failed to encode processed image to PNG buffer.")

        print("Detailed preprocessing complete.")
        return BytesIO(buf.tobytes())

    except Exception as e:
        print(f"Error during detailed image preprocessing: {e}")
        raise

def preprocess_image_minimal(image_path: str) -> BytesIO:
    """
    Minimal preprocessing: Read as grayscale, optional resize.
    Often a good starting point for advanced OCR services.
    """
    print(f"Preprocessing image (minimal): {image_path}")
    try:
        # Read the image directly as grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            else:
                raise IOError(f"Image file found but could not be read by OpenCV: {image_path}")

        # --- Optional: Save grayscale to inspect ---
        cv2.imwrite(f"{os.path.splitext(image_path)[0]}_preprocessed_minimal.png", image)

        # Optional Resize (Consider if originals are very small)
        min_dim = min(image.shape[:2])
        if min_dim < 600: # Lower threshold for minimal preprocessing
             scale_factor = 600 / min_dim
             if scale_factor > 1.0:
                 width = int(image.shape[1] * scale_factor)
                 height = int(image.shape[0] * scale_factor)
                 image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR) # Linear is faster
                 print(f"Resized image to {image.shape[1]}x{image.shape[0]}.")

        # Convert grayscale image to PNG buffer (lossless is good for OCR)
        is_success, buf = cv2.imencode(".png", image)
        if not is_success:
            raise IOError("Failed to encode processed image to PNG buffer.")

        print("Minimal preprocessing complete.")
        return BytesIO(buf.tobytes())

    except Exception as e:
        print(f"Error during minimal image preprocessing: {e}")
        raise

# --- OCR and Extraction ---

def extract_codes_from_words(words: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """
    Filter for potential alphanumeric codes from OCR words using word confidence.
    """
    # --- Adjust this Regex based on the specific format(s) of your codes ---
    # Example 1: Matches words that are 3+ chars long and contain only A-Z, 0-9
    # code_pattern = re.compile(r'^[A-Z0-9]{3,}$')
    # Example 2: Matches patterns like 'V1339E01' or 'A1339A' (Letter+Digits+Letter[+Digits])
    # code_pattern = re.compile(r'^[A-Z]+\d+[A-Z]+\d*$')
    # Example 3: Matches sequences of letters/digits possibly separated by single spaces (like '4B 339E 01')
    # This allows internal spaces but requires start/end with alphanumeric
    code_pattern = re.compile(r'^[A-Z0-9](?:[A-Z0-9\s]*[A-Z0-9])?$')
    # Example 4: A simpler version of Ex3, allowing any mix of 3+ letters, digits, spaces
    # code_pattern = re.compile(r'^[A-Z0-9\s]{3,}$') # Might be too broad

    # --- Choose ONE pattern or combine logic if needed ---
    # Let's use Example 3 as a starting point for the billet image provided
    print(f"Using regex pattern: {code_pattern.pattern}")

    filtered = []
    for text, conf in words:
        # Use fullmatch to ensure the *entire* word string matches the pattern
        # Remove potential leading/trailing spaces from OCR before matching
        cleaned_text = text.strip()
        if code_pattern.fullmatch(cleaned_text):
             # Optional: Add length constraints or other checks here if needed
             if len(cleaned_text) >= 3: # Example: Require minimum length
                 filtered.append((cleaned_text, conf))
        # else: # Optional: Debugging print for non-matches
             # print(f"  Word '{cleaned_text}' did not match pattern.")

    # Optional: Add post-filtering based on confidence threshold
    # min_confidence = 0.5 # Example threshold
    # filtered = [item for item in filtered if item[1] >= min_confidence]

    return filtered

# Replace the existing ocr_image function with this one:

from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes # Ensure this is imported
import traceback # Ensure traceback is imported

# You can add this temporarily right after imports to confirm the available members
# print("Available OperationStatusCodes members:", [member.value for member in OperationStatusCodes])
# Then remove it once confirmed.

def ocr_image(image_buffer: BytesIO) -> Tuple[str, List[Tuple[str, float]]]:
    """
    Call Azure OCR Read API and parse results to get words and their confidence.
    Includes robust handling for different terminal states (including 'canceled' string)
    and polling errors.
    """
    print("Calling Azure OCR Read API...")
    operation_id = None
    status: Any = "unknown_start"
    read_result = None # Initialize read_result
    words_conf: List[Tuple[str, float]] = []
    final_status_str: str = "unknown_end"

    try:
        image_buffer.seek(0)
        read_response = computervision_client.read_in_stream(image_buffer, raw=True)
        operation_location = read_response.headers["Operation-Location"]
        operation_id = operation_location.split("/")[-1]
        print(f"OCR operation ID: {operation_id}")

        max_wait_time = 120
        poll_interval = 1
        start_time = time.time()

        while True:
            current_time = time.time()
            if current_time - start_time > max_wait_time:
                 print(f"Warning: OCR operation timed out after {max_wait_time} seconds.")
                 status = "timedOut"
                 break

            try:
                read_result = computervision_client.get_read_result(operation_id)
                if not hasattr(read_result, 'status'):
                     print("  Warning: Read result object lacks 'status' attribute.")
                     status = "unknown_status_attribute_missing"
                     break

                status = read_result.status # Can be Enum or potentially string/None
                print(f"  OCR Status: {status} (Type: {type(status)})")

                # --- Check for Terminal States (including 'canceled' string) ---
                is_terminal = False
                status_str_value = "unknown" # Get string representation for checking

                if isinstance(status, OperationStatusCodes):
                    status_str_value = status.value # e.g., "succeeded", "failed", "running"
                    if status in [OperationStatusCodes.succeeded, OperationStatusCodes.failed]:
                        is_terminal = True
                    # NOTE: We know OperationStatusCodes.canceled caused error, so don't check it here.
                elif isinstance(status, str):
                     status_str_value = status.lower() # Handle potential case variations
                     # Check if the string indicates a known terminal state
                     if status_str_value in ["succeeded", "failed", "canceled"]:
                          is_terminal = True
                          print(f"  Detected terminal status via string: '{status_str_value}'")
                     elif status_str_value in ["running", "notstarted"]: # Allow known non-terminal strings too
                          pass # continue polling
                     else:
                          print(f"  Warning: Received unknown status string: '{status_str_value}'")
                          # Decide: Treat unknown string as terminal? Let's break loop.
                          status = f"unknown_unexpected_status_string_{status_str_value}"
                          is_terminal = True # Break loop on unknown string status

                else:
                     # Handle unexpected status types (None, int, etc.)
                     print(f"  Warning: Unexpected status type '{type(status)}'.")
                     status = f"unknown_unexpected_status_type_{type(status).__name__}"
                     is_terminal = True # Break loop

                if is_terminal:
                    break
                # --- End Terminal State Check ---

            except AttributeError as ae:
                 # Catch AttributeErrors specifically during polling if status object is weird
                 print(f"  AttributeError during polling status (Op ID: {operation_id}): {ae}")
                 print("  Traceback (polling AttributeError):")
                 traceback.print_exc()
                 status = f"error_polling_attributeerror_{ae}"
                 break # Exit loop
            except Exception as poll_error:
                 print(f"  ERROR during polling OCR status (Op ID: {operation_id}): {type(poll_error).__name__} - {poll_error}")
                 print("  Traceback (polling error):")
                 traceback.print_exc()
                 status = f"error_polling_{type(poll_error).__name__}"
                 break

            time.sleep(poll_interval)

        # --- Process the results based on the final status ---
        # Determine final status string robustly
        if isinstance(status, OperationStatusCodes):
            final_status_str = status.value
        else:
            final_status_str = str(status).lower() # Use lowercase string value

        print(f"Final determined status string: '{final_status_str}'") # Debug print

        # Process based on the final string status
        if final_status_str == "succeeded":
            print("OCR succeeded.")
            # Safely access analyze_result (ensure read_result is not None)
            if read_result and hasattr(read_result, 'analyze_result') and read_result.analyze_result and \
               hasattr(read_result.analyze_result, 'read_results') and read_result.analyze_result.read_results:
                for page in read_result.analyze_result.read_results:
                    if hasattr(page, 'lines') and page.lines:
                        for line in page.lines:
                            if hasattr(line, 'words') and line.words:
                                for word in line.words:
                                    conf = word.confidence if hasattr(word, 'confidence') and word.confidence is not None else 0.0
                                    words_conf.append((word.text.strip(), conf))
            # Check if words were actually extracted
            if not words_conf and final_status_str == "succeeded":
                 print("OCR status succeeded, but no words were extracted (result structure might be empty or missing details).")
                 final_status_str = "succeeded_no_text" # Update status for clarity downstream

        elif final_status_str == "failed":
             print(f"OCR operation failed.")
             error_message = "Unknown failure reason."
             # Attempt to get error details if available safely
             try:
                 if read_result and hasattr(read_result, 'analyze_result') and read_result.analyze_result and \
                    hasattr(read_result.analyze_result, 'errors') and read_result.analyze_result.errors:
                     err = read_result.analyze_result.errors[0]
                     error_message = f"Code: {err.code}, Message: {err.message}"
                 elif read_result and hasattr(read_result, 'message') and read_result.message:
                      error_message = read_result.message
             except Exception as e:
                 error_message = f"Could not parse specific error details: {type(e).__name__}"
             print(f"  Error: {error_message}")

        elif final_status_str == "canceled": # Check the STRING value
             print(f"OCR operation was canceled.")
             print(f"  Reason: Operation canceled by the service or potentially due to input/permission/quota issues.")

        else:
             # Handle other cases like timeout, polling errors, unknown status
             print(f"OCR did not succeed or finish normally. Final Status: {final_status_str}")
             if final_status_str.startswith("error_"):
                 print(f"  Reason: An error occurred during status polling or processing.")
             elif final_status_str == "timedout": # Ensure lowercase comparison
                 print(f"  Reason: Operation timed out.")
             else:
                 print(f"  Reason: Unknown or unexpected final status occurred.")

        print("OCR complete.")

    except Exception as e:
        # Catch critical errors during initial call or result processing setup
        print(f"CRITICAL Error during OCR call initiation or result processing (Op ID: {operation_id}): {type(e).__name__} - {e}")
        print("Traceback (critical error):")
        traceback.print_exc()
        final_status_str = f"error_critical_{type(e).__name__}"
        words_conf = [] # Ensure words_conf is empty

    # Always return the final status string and the (potentially empty) list of words
    return final_status_str, words_conf

# --- No changes needed below this line in the rest of the script ---
# The process_image function needs a small update to check the status string correctly

def process_image(image_path: str, preprocessing_mode: str = 'minimal') -> Dict[str, Any]:
    """
    Main function to process a single image file.
    Returns a dictionary containing processing results.
    'preprocessing_mode' can be 'minimal' or 'detailed'.
    """
    print(f"\n--- Processing: {image_path} (Mode: {preprocessing_mode}) ---")
    results: Dict[str, Any] = {
        "file_path": image_path,
        "status": "not_processed",
        "preprocessing_mode": preprocessing_mode,
        "preprocessing_status": "failed",
        "ocr_status": "failed",
        "all_ocr_words": [], # Store all detected words/conf
        "detected_codes": [], # Store filtered codes/conf
        "error": None
    }

    try:
        # Step 1: Preprocess the image
        # (Keep existing preprocessing logic call)
        if preprocessing_mode == 'detailed':
            image_buffer = preprocess_image_detailed(image_path)
        elif preprocessing_mode == 'minimal':
            image_buffer = preprocess_image_minimal(image_path)
        else:
             raise ValueError(f"Unknown preprocessing_mode: {preprocessing_mode}")
        results["preprocessing_status"] = "succeeded"


        # Step 2: Perform OCR on the processed image buffer
        ocr_status, ocr_words = ocr_image(image_buffer) # ocr_status is now always a string
        results["ocr_status"] = ocr_status
        results["all_ocr_words"] = ocr_words # Store all detected words for context/debugging

        # Step 3: Filter for detected codes if OCR succeeded
        # --- Updated Check ---
        # Check against the string "succeeded" or the specific "succeeded_no_text" status
        if ocr_status == "succeeded" or ocr_status == "succeeded_no_text":
             print("Filtering for codes...")
             detected_codes = extract_codes_from_words(ocr_words)

             # Sort by confidence (highest first)
             results["detected_codes"] = sorted(detected_codes, key=lambda x: x[1], reverse=True)
             results["status"] = "completed" if results["detected_codes"] else "completed_no_codes_found"

             print("\n--- Detected Codes (Sorted by Confidence) ---")
             if not results["detected_codes"]:
                  print(" No codes detected that match the pattern.")
             else:
                  for text, conf in results["detected_codes"]:
                      print(f"   Code: '{text}', Confidence: {conf:.4f}") # Higher precision for confidence
        else:
             # OCR did not succeed or timed out or other error
             results["status"] = f"ocr_{ocr_status}" # Use the actual status string
             print(f"Skipping code extraction due to OCR status: {ocr_status}")


    except FileNotFoundError as e:
        results["error"] = f"File not found: {e}"
        results["status"] = "failed_file_not_found"
        print(f"Error: {results['error']}")
    except ValueError as e: # Catch config errors like bad preprocessing mode
        results["error"] = f"Configuration Error: {e}"
        results["status"] = "failed_config_error"
        print(f"Error: {results['error']}")
    except IOError as e: # Catch image read/encode errors
        results["error"] = f"Image Processing Error: {e}"
        results["status"] = "failed_image_error"
        print(f"Error: {results['error']}")
    except Exception as e:
        results["error"] = f"An unexpected error occurred: {type(e).__name__} - {e}"
        results["status"] = "failed_unexpected"
        print(f"An unexpected error occurred during processing: {e}")
        # import traceback
        # traceback.print_exc()

    return results

# --- The rest of the script (preprocessing functions, extraction function, main execution block) remains the same ---
# --- Main Processing Logic ---

def process_image(image_path: str, preprocessing_mode: str = 'minimal') -> Dict[str, Any]:
    """
    Main function to process a single image file.
    Returns a dictionary containing processing results.
    'preprocessing_mode' can be 'minimal' or 'detailed'.
    """
    print(f"\n--- Processing: {image_path} (Mode: {preprocessing_mode}) ---")
    results: Dict[str, Any] = {
        "file_path": image_path,
        "status": "not_processed",
        "preprocessing_mode": preprocessing_mode,
        "preprocessing_status": "failed",
        "ocr_status": "failed",
        "all_ocr_words": [], # Store all detected words/conf
        "detected_codes": [], # Store filtered codes/conf
        "error": None
    }

    try:
        # Step 1: Preprocess the image
        if preprocessing_mode == 'detailed':
            image_buffer = preprocess_image_detailed(image_path)
        elif preprocessing_mode == 'minimal':
            image_buffer = preprocess_image_minimal(image_path)
        else:
             raise ValueError(f"Unknown preprocessing_mode: {preprocessing_mode}")
        results["preprocessing_status"] = "succeeded"

        # Step 2: Perform OCR on the processed image buffer
        ocr_status, ocr_words = ocr_image(image_buffer)
        results["ocr_status"] = ocr_status
        results["all_ocr_words"] = ocr_words # Store all detected words for context/debugging

        # Step 3: Filter for detected codes if OCR succeeded
        # Check against known success statuses
        if ocr_status == OperationStatusCodes.succeeded.value or ocr_status == "succeeded_no_text":
             print("Filtering for codes...")
             detected_codes = extract_codes_from_words(ocr_words)

             # Sort by confidence (highest first)
             results["detected_codes"] = sorted(detected_codes, key=lambda x: x[1], reverse=True)
             results["status"] = "completed" if results["detected_codes"] else "completed_no_codes_found"

             print("\n--- Detected Codes (Sorted by Confidence) ---")
             if not results["detected_codes"]:
                  print(" No codes detected that match the pattern.")
             else:
                  for text, conf in results["detected_codes"]:
                      print(f"   Code: '{text}', Confidence: {conf:.4f}") # Higher precision for confidence
        else:
             # OCR did not succeed or timed out or other error
             results["status"] = f"ocr_failed_or_issue: {ocr_status}"
             print(f"Skipping code extraction due to OCR status: {ocr_status}")


    except FileNotFoundError as e:
        results["error"] = f"File not found: {e}"
        results["status"] = "failed_file_not_found"
        print(f"Error: {results['error']}")
    except ValueError as e: # Catch config errors like bad preprocessing mode
        results["error"] = f"Configuration Error: {e}"
        results["status"] = "failed_config_error"
        print(f"Error: {results['error']}")
    except IOError as e: # Catch image read/encode errors
        results["error"] = f"Image Processing Error: {e}"
        results["status"] = "failed_image_error"
        print(f"Error: {results['error']}")
    except Exception as e:
        results["error"] = f"An unexpected error occurred: {type(e).__name__} - {e}"
        results["status"] = "failed_unexpected"
        print(f"An unexpected error occurred during processing: {e}")
        # Consider logging the full traceback here for debugging complex issues
        # import traceback
        # traceback.print_exc()

    return results


if __name__ == "__main__":
    # --- Configuration ---
    image_folder = "images" # Folder containing images relative to the script
    log_file_name = "ocr_results.txt" # Output log file
    # Choose preprocessing: 'minimal' or 'detailed'
    # Start with 'minimal' for textured surfaces, try 'detailed' if results are poor.
    selected_preprocessing_mode = 'minimal'
    # --- End Configuration ---

    # List of common image file extensions (case-insensitive)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

    print(f"Processing all images in folder: {image_folder}")
    print(f"Using preprocessing mode: {selected_preprocessing_mode}")
    print(f"Logging results to: {log_file_name}")

    # Ensure the images folder exists
    if not os.path.isdir(image_folder):
        print(f"Error: Folder not found at '{os.path.abspath(image_folder)}'")
        print("Please ensure the 'images' folder exists in the same directory as the script.")
        exit() # Exit if the folder doesn't exist

    # Open the log file for writing
    try:
        with open(log_file_name, "w", encoding="utf-8") as log_file:
            start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            log_file.write(f"OCR Results Log - Started: {start_time_str}\n")
            log_file.write(f"Preprocessing Mode Used: {selected_preprocessing_mode}\n")
            log_file.write("-" * 60 + "\n")

            found_images = False
            # Process each file in the folder
            for file_name in sorted(os.listdir(image_folder)): # Sort for consistent order
                full_file_path = os.path.join(image_folder, file_name)

                # Check if the path is a file and has a valid image extension
                if os.path.isfile(full_file_path):
                    file_extension = os.path.splitext(file_name)[1].lower()

                    if file_extension in image_extensions:
                        found_images = True
                        # Process the image file and get results
                        processing_results = process_image(
                            full_file_path,
                            preprocessing_mode=selected_preprocessing_mode
                        )

                        # Log the results
                        log_file.write(f"File: {processing_results['file_path']}\n")
                        log_file.write(f"Overall Status: {processing_results['status']}\n")
                        log_file.write(f"Preprocessing Status: {processing_results['preprocessing_status']}\n")
                        log_file.write(f"OCR Status: {processing_results['ocr_status']}\n")

                        if processing_results["error"]:
                            log_file.write(f"Error Details: {processing_results['error']}\n")

                        log_file.write("Detected Codes (Sorted by Confidence):\n")
                        if processing_results["detected_codes"]:
                            for text, conf in processing_results["detected_codes"]:
                                log_file.write(f"  Code: '{text}', Confidence: {conf:.4f}\n")
                        else:
                            log_file.write("  None found matching the pattern.\n")

                        # Optional: Log all detected words for debugging
                        log_file.write("All OCR Words (for reference):\n")
                        if processing_results["all_ocr_words"]:
                             # Limit logging if too verbose
                             limit = 50
                             for i, (text, conf) in enumerate(processing_results["all_ocr_words"]):
                                 if i < limit:
                                     log_file.write(f"  Word: '{text}', Confidence: {conf:.4f}\n")
                             if len(processing_results["all_ocr_words"]) > limit:
                                 log_file.write(f"  ... (truncated, total {len(processing_results['all_ocr_words'])} words)\n")
                        else:
                             log_file.write("  No words detected by OCR.\n")


                        log_file.write("-" * 30 + "\n\n") # Separator between files

                    else:
                        print(f"Skipping non-image file: {file_name}")
                        # Optionally log skipped files too
                        # log_file.write(f"File: {full_file_path}\n")
                        # log_file.write(f"Overall Status: Skipped (Not a recognized image file type)\n")
                        # log_file.write("-" * 30 + "\n\n")

            if not found_images:
                 log_file.write("No image files found in the specified folder.\n")
                 print(f"Warning: No image files ({', '.join(image_extensions)}) found in the '{image_folder}' directory.")

            end_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            log_file.write(f"\nOCR Results Log - Finished: {end_time_str}\n")
            log_file.write("-" * 60 + "\n")

        print(f"\n--- Finished processing. Results logged to {log_file_name} ---")

    except IOError as e:
        print(f"Error writing to log file {log_file_name}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during the main processing loop: {e}")
        # import traceback
        # traceback.print_exc() # Print detailed traceback for unexpected errors