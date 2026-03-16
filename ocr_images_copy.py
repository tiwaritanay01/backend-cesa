try:
    import pytesseract
except ImportError:
    pytesseract = None
    print("pytesseract not installed. OCR will be simulated.")
from PIL import Image
import os

def perform_ocr_on_images(image_paths):
    results = {}
    for path in image_paths:
        if os.path.exists(path):
            print(f"Processing {path}...")
            try:
                # Open image
                img = Image.open(path)
                # Perform OCR
                if pytesseract is None:
                    text = f"[Simulated OCR output for {path}]"
                else:
                    text = pytesseract.image_to_string(img)
                results[path] = text
            except Exception as e:
                results[path] = f"Error processing image: {str(e)}"
        else:
            results[path] = "File not found."
    return results

if __name__ == "__main__":
    images = [
        r"c:\Users\SACHIN JAISWAR\test_backend\gov-tech-main\WhatsApp Image 2026-03-05 at 11.40.34 AM.jpeg",
        r"c:\Users\SACHIN JAISWAR\test_backend\gov-tech-main\WhatsApp Image 2026-03-05 at 11.40.42 AM.jpeg"
    ]
    
    ocr_results = perform_ocr_on_images(images)
    
    output_file = "images_ocr_text.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for path, text in ocr_results.items():
            f.write(f"=== Results for: {path} ===\n")
            f.write(text)
            f.write("\n\n")
            
    print("\nOCR Complete! Details below:\n")
    for path, text in ocr_results.items():
        print(f"--- {path} ---")
        print(text[:500] + "..." if len(text) > 500 else text)
        print("-" * 20)
        
    print(f"\nFull text saved to {output_file}")
