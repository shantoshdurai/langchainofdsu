from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from pathlib import Path
import shutil

from chatbot import Chatbot

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

bot = Chatbot(
    model_name="llama3.1:8b",
    temperature=0.7,
    memory_window=10
)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ingest")
async def ingest(files: List[UploadFile] = File(default=[])):
    saved = []
    for f in files:
        dest = DATA_DIR / f.filename
        with dest.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(str(dest))
    all_docs = [str(p) for p in DATA_DIR.glob("*") if p.is_file()]
    if all_docs:
        bot.load_documents(all_docs)
        bot.setup_conversation_chain()
        return {"message": "Documents ingested", "files": saved}
    return {"message": "No documents in data/. Chat will run without RAG.", "files": saved}

@app.post("/chat")
async def chat(message: str = Form(...), session_id: Optional[str] = Form(default=None)):
    result = bot.chat(message)
    return result
