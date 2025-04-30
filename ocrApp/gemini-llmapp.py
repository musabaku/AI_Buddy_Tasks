import os
import re
import cv2
import numpy as np
from io import BytesIO
import base64
import time
from typing import List, Tuple, Dict, Any
import google.generativeai as genai
from tabulate import tabulate

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

# 1) Where you keep your images
IMAGE_FOLDER = "images"

# 2) Your Google GenAI API key (or use env var GENAI_API_KEY)
# API_KEY = os.getenv("GENAI_API_KEY", "")
API_KEY = os.getenv("GENAI_API_KEY", "")


# 3) Configure the GenAI library
genai.configure(api_key=API_KEY)


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
    if min_dim < 800:
        scale = 800.0 / min_dim
        new_w, new_h = int(w * scale), int(h * scale)
        thresh = cv2.resize(thresh, (new_w, new_h),
                            interpolation=cv2.INTER_LINEAR)

    success, buf = cv2.imencode('.jpg', thresh)
    if not success:
        raise IOError("Failed to JPEG-encode preprocessed image")
    return BytesIO(buf.tobytes())


def ocr_with_gemini(image_buf: BytesIO) -> Tuple[str, List[Tuple[str, float]]]:
    """
    Sends the image bytes to Gemini 2 for OCR.
    Returns a status and a list of (line_text, confidence).
    """
    try:
        image_bytes = image_buf.getvalue()
        # model = genai.GenerativeModel("gemini-2.0-pro-exp")
        # model = genai.GenerativeModel("gemini-2.0-pro-exp")
        # model = genai.GenerativeModel("gemini-2.0-pro-exp")
        model = genai.GenerativeModel("gemini-2.0-pro-exp-02-05")
        # model = genai.GenerativeModel("gemini-2.0-flash-thinking-exp")
        # model = genai.GenerativeModel("gemini-2.0-flash-thinking-exp-1219")

        contents = [
            {'mime_type': 'image/jpeg', 'data': image_bytes},
            "This is a close-up image of metal with engraved or embossed text in the center area. Please extract the extract alphanumeric sequence of characters (numbers and uppercase letters) as accurately as possible, preserving the order. Ignore surface texture or background artifacts. Do not add any commentary or formatting — only return the raw string or strings found."
        ]

        response = model.generate_content(contents, request_options={'timeout': 600})

        if not hasattr(response, 'text') or not response.text:
            if hasattr(response, 'candidates') and response.candidates:
                if hasattr(response.candidates[0], 'content') and response.candidates[0].content.parts:
                    print("Warning: API response missing .text attribute but has candidates.")
                    return "failed", []
            print(f"API response: {response}")
            return "failed", []

        text = response.text

        lines = [(line.strip(), 1.0)
                 for line in text.splitlines()
                 if line.strip()]

        return "succeeded", lines


    except Exception as e:
        print(f"An unexpected error occurred during OCR: {e}")
        return "failed_exception", []


def extract_numbers_from_text(lines: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """
    From the OCR’d lines, pull out any pure integers.
    """
    number_pattern = re.compile(r'\b\d+\b')
    results: List[Tuple[str, float]] = []
    for text, conf in lines:
        for m in number_pattern.findall(text):
            results.append((m, conf))
    return results


def process_image(image_path: str) -> Dict[str, Any]:
    """
    Full pipeline for one image:
      1) preprocess → BytesIO
      2) send to Gemini
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

        print("Sending to Gemini for OCR...")
        ocr_status, lines = ocr_with_gemini(buf)
        out["ocr_status"] = ocr_status
        out["all_ocr_lines"] = lines
        print(f"OCR status: {ocr_status}")

        if ocr_status == "succeeded":
            nums = extract_numbers_from_text(lines)
            out["detected_numbers"] = sorted(nums, key=lambda x: -x[1])
            out["status"] = "completed" if nums else "completed_no_numbers"
            if nums:
                print(f"Detected numbers: {nums}")
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
        note = result.get("error", "No issues" if detected != "-" else "No pure numbers found")

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

