# -------------------------------
# 🧠 SMART RAG ASSISTANT — Persistent + Hash-Based + Conversational Memory
# -------------------------------

from dotenv import load_dotenv
import os
from pathlib import Path
import hashlib

# Load environment variables
load_dotenv()

if os.getenv("GROQ_API_KEY"):
    print("✅ Groq API Key loaded:", os.getenv("GROQ_API_KEY")[:8], "********")
else:
    print("⚠️ GROQ_API_KEY not found! Please add it to your .env file.")

# -------------------------------
# LangChain imports
# -------------------------------
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# -------------------------------
# Step 1 — Helper: Compute file hash
# -------------------------------
def compute_md5(file_path):
    """Compute MD5 hash for a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

# -------------------------------
# Step 2 — Load PDFs & detect changes
# -------------------------------
data_folder = Path(__file__).parent / "data"
data_folder = data_folder.resolve()
chroma_dir = Path(__file__).parent / "chroma_db"
chroma_dir.mkdir(exist_ok=True)

print(f"📂 Scanning folder: {data_folder}")
pdf_files = list(data_folder.glob("*.pdf"))
print(f"📄 Found {len(pdf_files)} PDF files.")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Try loading existing ChromaDB
db = Chroma(persist_directory=str(chroma_dir), embedding_function=embeddings)

# Get existing file hashes
existing_data = db.get(include=["metadatas"])
existing_meta = existing_data.get("metadatas", [])
existing_hashes = {m["source"]: m["hash"] for m in existing_meta if "source" in m and "hash" in m}

print(f"🧠 Found {len(existing_hashes)} stored files in ChromaDB.")

# Compare with current files
to_process, to_delete = [], []

for file in pdf_files:
    file_hash = compute_md5(file)
    file_path = str(file)

    if file_path not in existing_hashes:
        print(f"🆕 New file: {file.name}")
        to_process.append((file, file_hash))
    elif existing_hashes[file_path] != file_hash:
        print(f"♻️ Updated file: {file.name}")
        to_process.append((file, file_hash))
        to_delete.append(file_path)
    else:
        print(f"✅ Unchanged: {file.name}")

# -------------------------------
# Step 3 — Update vector database
# -------------------------------
if to_delete:
    print(f"🗑 Removing outdated data for {len(to_delete)} modified files...")
    for f in to_delete:
        db.delete(where={"source": f})

if to_process:
    print(f"📥 Processing {len(to_process)} new/updated files...")
    all_docs = []
    for file, file_hash in to_process:
        loader = PyPDFLoader(str(file))
        docs = loader.load()
        for d in docs:
            d.metadata = {"source": str(file), "hash": file_hash}
        all_docs.extend(docs)

    print("✂️ Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(all_docs)

    print("🧮 Embedding and adding new documents to ChromaDB...")
    db.add_documents(chunks)
    db.persist()
    print("💾 ChromaDB updated successfully!")

else:
    print("✅ No new or modified files to process. Using existing ChromaDB.")

retriever = db.as_retriever(search_kwargs={"k": 3})

# -------------------------------
# Step 4 — LLM + Conversational Memory
# -------------------------------
llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=os.getenv("GROQ_API_KEY"))

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)

# -------------------------------
# Step 5 — Chat Loop
# -------------------------------
print("\n🤖 Smart RAG Assistant ready! Type your question (or 'exit' to quit).")

while True:
    query = input("\n🧑 You: ")
    if query.lower() in ["exit", "quit"]:
        print("👋 Goodbye! Session ended.")
        break

    result = qa.invoke({"question": query})

    print("\n💬 Assistant:", result["answer"])
