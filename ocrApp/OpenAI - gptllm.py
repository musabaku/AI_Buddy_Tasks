import os
import re
import cv2
import numpy as np
from io import BytesIO
import base64 # Import base64 for encoding image
import time
from typing import List, Tuple, Dict, Any
# Removed google.generativeai
# import google.generativeai as genai
from openai import OpenAI # Import OpenAI client
from tabulate import tabulate

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

# 1) Where you keep your images
IMAGE_FOLDER = "images"

OPENAI_API_KEY_HARDCODED = "-"

# 3) Initialize the OpenAI client with the hardcoded key
# Pass the api_key argument directly to the client constructor
client = OpenAI(api_key=OPENAI_API_KEY_HARDCODED)

# ------------------------------------------------------------------
# OCR Pipeline
# ------------------------------------------------------------------

def preprocess_image(image_path: str) -> BytesIO:
    """
    Load an image, convert to gray, denoise, threshold, and
    scale so that the smallest side is at least 800px.
    Returns a JPEG-in-memory buffer.
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    denoised = cv2.fastNlMeansDenoising(
        gray, None,
        h=40, templateWindowSize=7, searchWindowSize=21
    )

    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=41,
        C=5
    )

    h, w = thresh.shape
    min_dim = min(h, w)
    # Scaling is often not strictly necessary for modern vision models
    # but can be kept if you think it helps the preprocessing.
    # Let's keep it as it was in the original code.
    if min_dim < 800:
        scale = 800.0 / min_dim
        new_w, new_h = int(w * scale), int(h * scale)
        thresh = cv2.resize(thresh, (new_w, new_h),
                            interpolation=cv2.INTER_LINEAR)

    success, buf = cv2.imencode('.jpg', thresh)
    if not success:
        raise IOError("Failed to JPEG-encode preprocessed image")
    return BytesIO(buf.tobytes())


# Renamed function for clarity
def ocr_with_openai(image_buf: BytesIO) -> Tuple[str, List[Tuple[str, float]]]:
    """
    Sends the image bytes to OpenAI GPT-Vision for OCR.
    Returns a status and a list of (line_text, confidence).
    Note: OpenAI Vision API does not provide confidence scores per line,
    so confidence is set to 1.0.
    """
    try:
        image_bytes = image_buf.getvalue()
        # Encode the image bytes to base64
        base64_image = base64.b64encode(image_bytes).decode('utf-8')

        # Use a GPT model with Vision capabilities
        # gpt-4o is generally recommended if available and cost-effective for you
        # gpt-4-turbo is also an option (specifically gpt-4-turbo-2024-04-09 or gpt-4-vision-preview)
        GPT_MODEL = "o4-mini-2025-04-16" # You can change this to "gpt-4-turbo" or others

        # Define the prompt as part of the messages content
        prompt_text = "This is a close-up image of metal with engraved or embossed text in the center area. Please extract the exact alphanumeric sequence of characters (numbers and uppercase letters) as accurately as possible, preserving the order. Ignore surface texture or background artifacts. Do not add any commentary or formatting — only return the raw string or strings found."

        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                                # You can add detail: "low" or "high" for potentially higher cost/better detail
                                # "detail": "auto" # auto, low, or high
                            },
                        },
                    ],
                }
            ],
            max_completion_tokens=500, # Limit the response length to prevent verbosity
        )

        # Check for the response text
        if not response.choices or not response.choices[0].message or not response.choices[0].message.content:
             print(f"API response: {response}")
             # Check if there's a reason for failure in the response, e.g., content filtering
             if response.choices and response.choices[0].finish_reason:
                  print(f"Finish reason: {response.choices[0].finish_reason}")
             return "failed", []


        text = response.choices[0].message.content.strip()

        # Split text into lines. OpenAI might return a single string or multiple lines.
        # We assume each line returned by the model is a separate piece of text to consider.
        # Assign a confidence of 1.0 as OpenAI does not provide per-line confidence.
        lines = [(line.strip(), 1.0)
                 for line in text.splitlines()
                 if line.strip()]

        if not lines:
             # If the model returned nothing after stripping whitespace
             print("Warning: OpenAI model returned empty or whitespace-only response.")
             return "succeeded_no_text", [] # Indicate success but no text found

        return "succeeded", lines

    except Exception as e:
        print(f"An unexpected error occurred during OCR: {e}")
        # You might want to log the specific type of OpenAI error if needed
        # from openai import APIError, APIConnectionError, RateLimitError
        # if isinstance(e, APIError): print(f"OpenAI API Error: {e.status_code} - {e.response}")
        return "failed_exception", []


def extract_numbers_from_text(lines: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """
    From the OCR’d lines, pull out any pure integers.
    """
    number_pattern = re.compile(r'\b\d+\b')
    results: List[Tuple[str, float]] = []
    for text, conf in lines: # Conf is always 1.0 now
        for m in number_pattern.findall(text):
            # Keep the confidence value (which is always 1.0) for consistency
            results.append((m, conf))
    return results


def process_image(image_path: str) -> Dict[str, Any]:
    """
    Full pipeline for one image:
      1) preprocess → BytesIO
      2) send to OpenAI
      3) parse out numbers
    Returns a dict with status, all OCR lines, and any numbers.
    """
    out: Dict[str, Any] = {
        "file_path": image_path,
        "status": "not_processed",
        "ocr_status": "failed",
        "all_ocr_lines": [],
        "detected_numbers": []
    }

    try:
        print(f"Processing {image_path}...")
        buf = preprocess_image(image_path)
        print("Preprocessing complete.")

        print("Sending to OpenAI for OCR...")
        # Call the renamed function
        ocr_status, lines = ocr_with_openai(buf)
        out["ocr_status"] = ocr_status
        out["all_ocr_lines"] = lines
        print(f"OCR status: {ocr_status}")

        # Check if OCR succeeded or succeeded but found no text initially
        if ocr_status.startswith("succeeded"):
            nums = extract_numbers_from_text(lines)
            # Sorting by confidence is less meaningful now as it's always 1.0
            out["detected_numbers"] = sorted(nums, key=lambda x: -x[1]) # Still sort for consistency

            if nums:
                 out["status"] = "completed"
                 print(f"Detected numbers: {nums}")
            else:
                 # Distinguish between OCR failure and OCR success but no numbers found
                 if ocr_status == "succeeded_no_text":
                      out["status"] = "completed_no_ocr_text"
                 else:
                      out["status"] = "completed_no_numbers"
                 print("No pure numbers detected.")
        else:
            out["status"] = f"ocr_failed_{ocr_status}"
            print(f"OCR failed for {image_path} with status: {ocr_status}")

    except FileNotFoundError as e:
        out["status"] = "error_file_not_found"
        out["error"] = str(e)
        print(f"Error processing {image_path}: {e}")
    except IOError as e:
        out["status"] = "error_image_processing"
        out["error"] = str(e)
        print(f"Error processing {image_path}: {e}")
    except Exception as e:
        out["status"] = "error_unexpected"
        out["error"] = str(e)
        print(f"An unexpected error occurred processing {image_path}: {e}")

    return out


# ------------------------------------------------------------------
# Main: process every file in IMAGE_FOLDER
# ------------------------------------------------------------------

if __name__ == "__main__":
    start = time.time()
    folder = IMAGE_FOLDER

    if not os.path.exists(folder):
        print(f"Error: Image folder '{folder}' not found.")
        exit()

    image_files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

    if not image_files:
        print(f"No files found in '{folder}'. Please add images to process.")


    # Store all results
    all_results = []

    for fname in image_files:
        path = os.path.join(folder, fname)
        result = process_image(path)

        detected = ", ".join([num for num, _ in result['detected_numbers']]) if result['detected_numbers'] else "-"

        # Adjust note based on new status types
        note = result.get("error", "") # Start with potential error message
        if not note: # If no error, set note based on status and detected numbers
            if result['status'] == 'completed':
                 note = "Numbers found"
            elif result['status'] == 'completed_no_numbers':
                 note = "OCR successful, but no pure numbers found"
            elif result['status'] == 'completed_no_ocr_text':
                 note = "OCR successful, but no text extracted by model"
            elif result['status'].startswith('ocr_failed'):
                 note = f"OCR failed: {result['ocr_status']}"
            else: # Should cover other error_ statuses
                 note = f"Processing error: {result['status']}"


        all_results.append([
            os.path.basename(result['file_path']),
            result['status'],
            result['ocr_status'],
            detected,
            note
        ])

    # Print tabular log
    print("\n📋 OCR Processing Report:\n")
    headers = ["File Name", "Overall Status", "OCR Status", "Detected Numbers", "Notes"]
    print(tabulate(all_results, headers=headers, tablefmt="grid"))

    elapsed = time.time() - start
    print(f"\n⏱ Total processing time: {elapsed:.1f}s")