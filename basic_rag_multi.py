# -------------------------------
# 🧠 STEP 3 — RAG SYSTEM FOR MULTIPLE PDFs
# -------------------------------

from dotenv import load_dotenv
import os
from pathlib import Path

# Load environment variables
load_dotenv()

# Check Groq API key
if os.getenv("GROQ_API_KEY"):
    print("Groq API Key loaded:", os.getenv("GROQ_API_KEY")[:8], "********")
else:
    print("⚠️ GROQ_API_KEY not found! Please add it to your .env file.")

# -------------------------------
# Imports for LangChain
# -------------------------------
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

# -------------------------------
# Step 1 — Load ALL PDFs from data folder
# -------------------------------
data_folder = Path(__file__).parent / "data"
data_folder = data_folder.resolve()

print(f"📂 Loading all PDFs from: {data_folder}")

# Automatically load every PDF file in the folder
loader = DirectoryLoader(str(data_folder), glob="**/*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

print(f"✅ Loaded {len(documents)} pages from all PDFs combined.")

# -------------------------------
# Step 2 — Split text into chunks
# -------------------------------
print("🔹 Splitting text into manageable chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(documents)
print(f"✅ Split into {len(chunks)} text chunks from all files.")

# -------------------------------
# Step 3 — Create embeddings (HuggingFace)
# -------------------------------
print("🧠 Creating embeddings for all chunks...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Store embeddings in FAISS vector DB
db = FAISS.from_documents(chunks, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 3})

# -------------------------------
# Step 4 — Connect to Groq LLM
# -------------------------------
llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=os.getenv("GROQ_API_KEY"))

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# -------------------------------
# Step 5 — Ask Questions Across PDFs
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
    print("💬 ANSWER:", result['result'])
    print("-----------------------------")

    # Show which files and pages the answer came from
    for i, doc in enumerate(result["source_documents"], start=1):
        print(f"\n📄 Source {i}: {doc.metadata['source']} (Page {doc.metadata.get('page', '?')})")
        print(doc.page_content[:200], "...")
