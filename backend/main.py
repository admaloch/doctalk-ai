from fastapi import FastAPI, UploadFile, File
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()  # Loads from .env file
import supabase
from supabase import create_client, Client  # <-- Make sure this is imported
import os
import io
import requests  # <-- ADD THIS
import json      # <-- ADD THIS

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "DocTalk AI Backend is Liverrr!"}

load_dotenv()  # Loads from .env file

# Ollama configuration - ADD THIS
OLLAMA_URL = "http://localhost:11434"

# Then your Supabase client
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

@app.get("/test-db")
def test_db():
    try:
        # Simple test query
        response = supabase.table('documents').select("*").limit(1).execute()
        return {"status": "Database connected!", "data": response.data}
    except Exception as e:
        return {"error": str(e)}


# Embedding function - ADD THIS
def get_embedding(text: str) -> list:
    """Get embedding vector from Ollama"""
    payload = {
        "model": "nomic-embed-text", 
        "prompt": text
    }
    response = requests.post(f"{OLLAMA_URL}/api/embeddings", json=payload)
    response.raise_for_status()
    return response.json()["embedding"]

# The core PDF processing pipeline
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    print(f"📥 Received file: {file.filename}")
    
    # 1. Read PDF
    contents = await file.read()
    
    # 2. Extract text with pypdf
    pdf_text = ""
    reader = PdfReader(io.BytesIO(contents))
    for page in reader.pages:
        pdf_text += page.extract_text() + "\n"
    
    print(f"📄 Extracted {len(pdf_text)} characters of text")
    
    # 3. Chunk text with LangChain
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(pdf_text)
    
    print(f"✂️ Split into {len(chunks)} chunks")
    
    # 4. Generate embeddings for first 2 chunks (for testing) - ADD THIS SECTION
    print("🧠 Generating embeddings...")
    for i, chunk in enumerate(chunks[:2]):  # Test with just 2 chunks first
        embedding = get_embedding(chunk)
        print(f"   Chunk {i+1}: Generated embedding with {len(embedding)} dimensions")
        print(f"   First 5 values: {embedding[:5]}...")
    
    return {
        "filename": file.filename,
        "text_length": len(pdf_text),
        "chunk_count": len(chunks),
        "embedding_test": f"Generated embeddings for first 2 chunks - each has {len(embedding)} dimensions"  # UPDATE THIS
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)