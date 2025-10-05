# -------------------------------
# 🧠 SMART PERSISTENT RAG (HASH-BASED UPDATE DETECTION)
# -------------------------------

from dotenv import load_dotenv
import os
from pathlib import Path
import hashlib

# Load environment variables
load_dotenv()

if os.getenv("GROQ_API_KEY"):
    print("Groq API Key loaded:", os.getenv("GROQ_API_KEY")[:8], "********")
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
from langchain.chains import RetrievalQA

# -------------------------------
# Step 1 — Helper: Compute file hash (MD5)
# -------------------------------
def compute_md5(file_path):
    """Compute MD5 hash of file content."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

# -------------------------------
# Step 2 — Load PDFs and check for updates
# -------------------------------
data_folder = Path(__file__).parent / "data"
data_folder = data_folder.resolve()

print(f"📂 Scanning folder: {data_folder}")
pdf_files = list(data_folder.glob("*.pdf"))
print(f"📄 Found {len(pdf_files)} PDF files.")

chroma_dir = Path(__file__).parent / "chroma_db"
chroma_dir.mkdir(exist_ok=True)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load existing DB (if any)
db = Chroma(persist_directory=str(chroma_dir), embedding_function=embeddings)

# Retrieve existing metadata
existing_data = db.get(include=["metadatas"])
existing_meta = existing_data.get("metadatas", [])

existing_file_info = {}
for meta in existing_meta:
    source = meta.get("source")
    file_hash = meta.get("hash")
    if source and file_hash:
        existing_file_info[source] = file_hash

print(f"🧠 Found {len(existing_file_info)} files already stored in ChromaDB.")

# Determine new or updated files
to_process = []
to_delete = []

for file in pdf_files:
    file_hash = compute_md5(file)
    file_path_str = str(file)

    if file_path_str not in existing_file_info:
        print(f"🆕 New file detected: {file.name}")
        to_process.append((file, file_hash))
    elif existing_file_info[file_path_str] != file_hash:
        print(f"♻️ File updated: {file.name}")
        to_process.append((file, file_hash))
        to_delete.append(file_path_str)
    else:
        print(f"✅ Unchanged: {file.name}")

# -------------------------------
# Step 3 — Update the ChromaDB
# -------------------------------
if to_delete:
    print(f"🗑 Removing outdated entries for {len(to_delete)} modified files...")
    for f in to_delete:
        db.delete(where={"source": f})

if to_process:
    print(f"📥 Processing {len(to_process)} new/updated files...")
    all_docs = []

    for file, file_hash in to_process:
        loader = PyPDFLoader(str(file))
        docs = loader.load()

        # Add metadata (file + hash)
        for d in docs:
            d.metadata = {"source": str(file), "hash": file_hash}

        all_docs.extend(docs)

    print("✂️ Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(all_docs)

    print("🧮 Creating embeddings and adding to ChromaDB...")
    db.add_documents(chunks)
    db.persist()
    print("💾 Database updated successfully!")

else:
    print("✅ No new or modified files to process. Using existing ChromaDB.")

# -------------------------------
# Step 4 — Retrieval + LLM setup
# -------------------------------
retriever = db.as_retriever(search_kwargs={"k": 3})
llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=os.getenv("GROQ_API_KEY"))

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# -------------------------------
# Step 5 — Interactive QA
# -------------------------------
while True:
    query = input("\n🤖 Ask a question (or type 'exit' to quit): ")
    if query.lower() in ["exit", "quit"]:
        print("👋 Goodbye!")
        break

    result = qa.invoke({"query": query})

    print("\n-----------------------------")
    print("🔍 QUESTION:", query)
    print("-----------------------------")
    print("💬 ANSWER:", result["result"])
    print("-----------------------------")

    for i, doc in enumerate(result["source_documents"], start=1):
        print(f"\n📄 Source {i}: {Path(doc.metadata['source']).name}")
        print(doc.page_content[:200], "...")
