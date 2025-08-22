import pytesseract
from PIL import Image


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

image_path = "sample_pid.png"   
try:
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)

    print("✅ Extracted Text:")
    print(text)

except FileNotFoundError:
    print(f"❌ Image not found at path: {image_path}")
