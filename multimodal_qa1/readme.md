# 🧠 Multimodal Data Processing System (Streamlit + Gemini)

This project is a **multimodal knowledge base system** built with **Streamlit**, **Gemini (Google Generative AI)**, and **SQLite**.  
It allows you to upload and process files of different types — text, PDFs, images, audio, and YouTube videos — extract their text, embed them for semantic search, and query the knowledge base with natural language questions.

---

## 🚀 Features

✅ Extracts text from:
- `.pdf`, `.docx`, `.txt`, `.md`
- `.jpg`, `.png` (OCR using Tesseract)
- `.mp3`, `.mp4`, `.wav` (via Whisper)
- YouTube videos (via `youtube_transcript_api`)

✅ Stores data in an **SQLite** database  
✅ Generates **text embeddings** using `sentence-transformers` + FAISS  
✅ Supports **semantic search**  
✅ Integrates **Google Gemini API** for natural-language answers  
✅ Modular design (each feature in its own helper module)

---

 install dependeices for appy
pip install -r requirements.txt
sudo apt install tesseract-ocr

Need to create gemini keys 
GEMINI_API_KEY=your_google_gemini_api_key_here
🔗 https://aistudio.google.com/app/apikey
pip install sentence-transformers faiss-cpu
pip install pdfplumber python-docx pytesseract pillow

sudo apt install tesseract-ocr  # (Linux)
pip install streamlit sqlalchemy pillow pdfplumber python-docx pytesseract pydub moviepy youtube-transcript-api google-generativeai sentence-transformers faiss-cpu openai-whisper
streamlit run app.py
