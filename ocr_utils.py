import pytesseract
from PIL import Image
import pypdfium2 as pdfium
import io
import os
import google.generativeai as genai

# Tesseract configuration
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def pdf_to_images(pdf_content):
    """
    Converts PDF byte content to a list of PIL images.
    """
    pdf = pdfium.PdfDocument(pdf_content)
    images = []
    for i in range(len(pdf)):
        page = pdf.get_page(i)
        bitmap = page.render(
            scale=2,  # Increase scale for better OCR quality
            rotation=0,
        )
        pil_image = bitmap.to_pil()
        images.append(pil_image)
    return images

def perform_ocr_with_gemini(content, filename, api_key):
    """
    Reliable OCR extraction using Gemini flash.
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Determine mime type
        ext = os.path.splitext(filename)[1].lower()
        mime_type = "application/pdf" if ext == ".pdf" else "image/jpeg"
        if ext in [".png"]: mime_type = "image/png"
        
        prompt = "Extract all readable text from this document accurately. Output the raw text only."
        
        response = model.generate_content([
            prompt,
            {"mime_type": mime_type, "data": content}
        ])
        
        return response.text
    except Exception as e:
        print(f"Gemini OCR Fallback Error: {e}")
        return None

def perform_ocr_on_pdf(pdf_content):
    """
    Performs OCR on a PDF file content.
    """
    images = pdf_to_images(pdf_content)
    
    full_text = ""
    for i, image in enumerate(images):
        try:
            text = pytesseract.image_to_string(image)
        except pytesseract.pytesseract.TesseractNotFoundError:
            return None # Signal failure to use fallback
        full_text += f"--- Page {i+1} ---\n{text}\n\n"
        
    return full_text

def perform_ocr_on_image(image_content):
    """
    Performs OCR on an image byte content.
    """
    img = Image.open(io.BytesIO(image_content))
    try:
        text = pytesseract.image_to_string(img)
    except pytesseract.pytesseract.TesseractNotFoundError:
        return None # Signal failure to use fallback
    return text


def process_document(filename, content, api_key=None):
    """
    Dispatches to the correct OCR function based on file extension.
    Uses Tesseract primarily, but falls back to Gemini if Tesseract is missing.
    """
    ext = os.path.splitext(filename)[1].lower()
    ocr_result = None
    
    if ext == ".pdf":
        ocr_result = perform_ocr_on_pdf(content)
    elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
        ocr_result = perform_ocr_on_image(content)
        
    # If Tesseract failed (returned None) and we have an API key, use Gemini for REAL OCR
    if ocr_result is None and api_key:
        print(f"Tesseract not found. Falling back to Gemini for real OCR on {filename}...")
        ocr_result = perform_ocr_with_gemini(content, filename, api_key)
    
    # Absolute final fallback if both fail
    if ocr_result is None:
        return f"[Error] Could not perform real OCR. Tesseract is missing and AI fallback failed."
        
    return ocr_result
