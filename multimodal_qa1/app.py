# app.py
import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv

import streamlit as st
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import sessionmaker, declarative_base

# Try to import the helper modules you uploaded. If a function isn't found, use a small fallback.
try:
    from extract_text import extract_text as module_extract_text
except Exception:
    module_extract_text = None

try:
    from audio_transcribe import transcribe_audio as module_transcribe_audio
except Exception:
    module_transcribe_audio = None

try:
    from chunker import chunk_text as module_chunk_text
except Exception:
    module_chunk_text = None

try:
    # expected functions: get_embeddings(texts)->list[list[float]], upsert_embeddings(id, texts, embeddings),
    # similarity_search(query_embedding, top_k)-> list[(id, score, text)]
    from embeddings import get_embeddings as module_get_embeddings, \
                           upsert_embeddings as module_upsert_embeddings, \
                           similarity_search as module_similarity_search
except Exception:
    module_get_embeddings = None
    module_upsert_embeddings = None
    module_similarity_search = None

# Load env
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.warning("GEMINI_API_KEY is not set in the environment. Set it in your .env file for Gemini responses.")

# Only import google generative if API key present to avoid runtime errors
if GEMINI_API_KEY:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)


# --- Database setup (unchanged / similar to your snippet) ---
Base = declarative_base()
engine = create_engine("sqlite:///knowledge_base.db", echo=False)
Session = sessionmaker(bind=engine)
session = Session()


class FileData(Base):
    __tablename__ = "files"
    id = Column(Integer, primary_key=True)
    filename = Column(String(255))
    filetype = Column(String(50))
    content = Column(Text)


Base.metadata.create_all(engine)


def save_to_db(filename, filetype, content):
    record = FileData(filename=filename, filetype=filetype, content=content)
    session.add(record)
    session.commit()
    return record.id


def get_all_records():
    return session.query(FileData).all()


# --- Fallbacks for missing helper functions --- #
def fallback_extract_text(uploaded_file):
    """
    Basic fallback: try reading as text; for binary fall back to a placeholder.
    You're encouraged to use your extract_text.py for better handling (pdf/docx/audio/images).
    """
    name = uploaded_file.name.lower()
    try:
        if name.endswith((".txt", ".md")):
            return uploaded_file.read().decode("utf-8")
        else:
            return "[Extraction not supported for this file type in fallback. Please use extract_text.py]"
    except Exception:
        return "[Failed to extract text - fallback]"


def fallback_transcribe_audio(path_or_file):
    # Very naive placeholder (use your audio_transcribe.py for real transcription)
    return "[Audio transcription placeholder - install and use a real transcriber]" 


def fallback_chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def fallback_get_embeddings(texts):
    # Dummy embeddings (NOT useful for similarity). Replace by your real embeddings module.
    return [[float(len(t))] for t in texts]


def fallback_similarity_search(query_embedding, top_k=3):
    # fallback: return top recent records (no semantic similarity)
    recs = session.query(FileData).order_by(FileData.id.desc()).limit(top_k).all()
    return [(r.id, 0.0, r.content) for r in recs]


# Choose implementations (prefer module functions if available)
extract_text = module_extract_text or fallback_extract_text
transcribe_audio = module_transcribe_audio or fallback_transcribe_audio
chunk_text = module_chunk_text or fallback_chunk_text
get_embeddings = module_get_embeddings or fallback_get_embeddings
upsert_embeddings = module_upsert_embeddings  # optional; may be None
similarity_search = module_similarity_search or fallback_similarity_search


# --- Gemini helper --- #
def ask_gemini(prompt, model_name="gemini-pro"):
    if not GEMINI_API_KEY:
        return "[Gemini API key not configured: cannot ask model]"
    model = genai.GenerativeModel(model_name)
    # Using a simple generate_content call similar to your snippet
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"[Gemini request failed: {e}]"


# --- Streamlit UI --- #
st.set_page_config(page_title="Multimodal Data Processing", layout="centered")
st.title("🧠 Multimodal Data Processing System (modular)")

uploaded_file = st.file_uploader("Upload a file (pdf, docx, txt, jpg, png, mp3, mp4)", type=["pdf", "docx", "txt", "jpg", "png", "mp3", "mp4"])
youtube_url = st.text_input("Or paste a YouTube URL (optional)")

with st.expander("Advanced / debug"):
    st.write("Using helper modules (detected):")
    st.write({
        "extract_text": bool(module_extract_text),
        "audio_transcribe": bool(module_transcribe_audio),
        "chunker": bool(module_chunk_text),
        "embeddings": bool(module_get_embeddings),
        "upsert_embeddings": bool(module_upsert_embeddings),
        "similarity_search": bool(module_similarity_search),
    })


if uploaded_file or youtube_url:
    st.info("Processing...")
    extracted_text = ""

    # If file is uploaded - prefer using the extract_text helper
    if uploaded_file:
        # If audio file types - handle as audio
        fname = uploaded_file.name.lower()
        if fname.endswith((".mp3", ".wav", ".m4a", ".aac")):
            # Save temp file (some helper transcribers expect a path)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(fname).suffix)
            tmp.write(uploaded_file.read())
            tmp.flush()
            tmp.close()
            try:
                # prefer module transcribe_audio(path) if available
                extracted_text = transcribe_audio(tmp.name) if module_transcribe_audio else fallback_transcribe_audio(tmp.name)
            finally:
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass
        else:
            # Non-audio file: use extract_text helper
            try:
                # If the module_extract_text expects a file-like object, pass uploaded_file directly.
                extracted_text = extract_text(uploaded_file)
            except TypeError:
                # fallback: pass path-like (write to temp)
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix)
                tmp.write(uploaded_file.read())
                tmp.flush()
                tmp.close()
                try:
                    extracted_text = extract_text(tmp.name)
                finally:
                    try:
                        os.unlink(tmp.name)
                    except Exception:
                        pass

        record_id = save_to_db(uploaded_file.name, uploaded_file.type, extracted_text)

    else:
        # youtube url processing placeholder: if you have extract_text.youtube function use that
        try:
            # attempt to use a function named 'extract_youtube_text' in extract_text module
            from extract_text import extract_youtube_text as module_extract_youtube
            extracted_text = module_extract_youtube(youtube_url)
        except Exception:
            # fallback: store the url and short note
            extracted_text = "[YouTube transcript extraction not available in this environment]"
        record_id = save_to_db(youtube_url, "youtube", extracted_text)

    st.subheader("📄 Extracted Data")
    st.text_area("Extracted Text", extracted_text[:3000], height=250)

    # Chunk & index text (if embeddings module exists, create embeddings)
    st.subheader("🧩 Chunking + Embedding (optional)")

    chunks = chunk_text(extracted_text) if extracted_text else []
    st.write(f"Created {len(chunks)} chunks (first 3 shown):")
    for i, c in enumerate(chunks[:3]):
        st.write(f"Chunk {i+1} length={len(c)}")
        st.text_area(f"chunk_{i+1}", c, height=120)

    if chunks and get_embeddings:
        st.info("Generating embeddings...")
        try:
            embeddings = get_embeddings(chunks)
            # optionally upsert to vector DB via your embeddings module
            if upsert_embeddings:
                # create a simple unique id per chunk using record id
                for i, chunk in enumerate(chunks):
                    upsert_embeddings(f"{record_id}_{i}", chunk, embeddings[i])
                st.success("Embeddings upserted to vector store (via embeddings.upsert_embeddings).")
            else:
                st.success("Embeddings created (module_get_embeddings provided) but upsert function not available.")
        except Exception as e:
            st.error(f"Failed to create embeddings: {e}")
    elif not chunks:
        st.info("No chunks to embed.")
    else:
        st.info("Embeddings module not available; skipping embedding generation.")


st.divider()

st.subheader("🔍 Query the Knowledge Base")
user_query = st.text_input("Ask a question about uploaded content")

if st.button("Search"):
    if not user_query:
        st.warning("Enter a question first.")
    else:
        # Compose context using semantic search OR naive fallback
        context_snippets = []
        if module_get_embeddings and module_similarity_search:
            # create query embedding
            try:
                q_emb = get_embeddings([user_query])[0]
                hits = similarity_search(q_emb, top_k=3)
                # hits expected as list[(id, score, text)] or list[(text, score)]
                for h in hits:
                    if len(h) == 3:
                        _, score, text = h
                    elif len(h) == 2:
                        text, score = h
                    else:
                        text = str(h)
                    context_snippets.append(text)
            except Exception as e:
                st.error(f"Semantic search failed: {e}")
        else:
            # fallback: simple DB retrieval + keyword filter
            recs = get_all_records()
            # rank by simple substring count
            ranked = sorted(recs, key=lambda r: user_query.lower() in (r.content or "").lower(), reverse=True)
            for r in ranked[:3]:
                context_snippets.append(r.content)

        if not context_snippets:
            st.info("No context found in DB. Showing entire DB as fallback.")
            context_snippets = [f"{r.filename}:\n{r.content}" for r in get_all_records()]

        final_prompt = (
            "You are an assistant that answers questions based on the provided knowledge base context.\n\n"
            "Context:\n"
            + "\n\n---\n\n".join(context_snippets)
            + f"\n\nUser question: {user_query}\n\nAnswer concisely in natural language, and cite context lines when helpful."
        )

        st.write("Prompt sent to Gemini (truncated):")
        st.code(final_prompt[:2000] + ("...\n\n(Truncated)" if len(final_prompt) > 2000 else ""))

        answer = ask_gemini(final_prompt)
        st.subheader("Answer from Gemini")
        st.write(answer)


# show recent DB records
st.divider()
st.subheader("📚 Knowledge Base Records (recent)")
records = get_all_records()
if records:
    for r in records[::-1][:10]:
        st.markdown(f"**{r.filename}** — _{r.filetype}_ (id: {r.id})")
        st.write(r.content[:400])
        st.write("---")
else:
    st.write("No records yet. Upload a file to get started.")
