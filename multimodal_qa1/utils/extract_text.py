# extract_text.py
import pdfplumber
import pytesseract
from PIL import Image
from docx import Document

def extract_text(file):
    """
    Extract text from various file types.
    file can be a path or file-like object.
    """
    import io, os
    text = ""
    name = file.name.lower() if hasattr(file, "name") else str(file).lower()

    try:
        if name.endswith(".pdf"):
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""

        elif name.endswith(".docx"):
            doc = Document(file)
            text = "\n".join([p.text for p in doc.paragraphs])

        elif name.endswith((".txt", ".md")):
            if isinstance(file, io.BytesIO):
                text = file.read().decode("utf-8")
            else:
                with open(file, "r", encoding="utf-8") as f:
                    text = f.read()

        elif name.endswith((".jpg", ".jpeg", ".png")):
            image = Image.open(file)
            text = pytesseract.image_to_string(image)

        else:
            text = "[Unsupported file type]"

    except Exception as e:
        text = f"[Error extracting text: {e}]"

    return text
