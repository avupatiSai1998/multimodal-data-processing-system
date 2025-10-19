import shutil
import os

# Create folder structure for project
os.makedirs("multimodal_qa/utils", exist_ok=True)

# ----------------- APP.PY -----------------
app_code = r'''[PUT THE FINAL APP CODE FROM OUR CHAT HERE]'''

# ----------------- EXTRACT_TEXT.PY -----------------
extract_text_code = r'''[PUT THE EXTRACT_TEXT CODE HERE]'''

# ----------------- AUDIO_TRANSCRIBE.PY -----------------
audio_transcribe_code = r'''[PUT THE AUDIO_TRANSCRIBE CODE HERE]'''

# ----------------- EMBEDDINGS.PY -----------------
embeddings_code = r'''[PUT THE EMBEDDINGS CODE HERE]'''

# ----------------- CHUNKER.PY -----------------
chunker_code = r'''[PUT THE CHUNKER CODE HERE]'''

# ----------------- README -----------------
readme_code = r'''[PUT THE README FROM OUR CHAT HERE]'''

# ----------------- REQUIREMENTS.TXT -----------------
requirements_code = '''streamlit
faiss-cpu
sentence-transformers
google-generativeai
PyPDF2
python-docx
pdfplumber
pytesseract
pillow
whisper
pytube
ffmpeg-python
SpeechRecognition
'''

# Write all files
files = {
    "multimodal_qa/app.py": app_code,
    "multimodal_qa/utils/extract_text.py": extract_text_code,
    "multimodal_qa/utils/audio_transcribe.py": audio_transcribe_code,
    "multimodal_qa/utils/embeddings.py": embeddings_code,
    "multimodal_qa/utils/chunker.py": chunker_code,
    "multimodal_qa/README.md": readme_code,
    "multimodal_qa/requirements.txt": requirements_code
}

for path, content in files.items():
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

# Create zip archive
shutil.make_archive("multimodal_qa", 'zip', "multimodal_qa")
print("✅ Project zipped as multimodal_qa.zip")

