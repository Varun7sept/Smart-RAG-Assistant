# # -------------------------------
# # 🧠 100% FREE RAG SYSTEM USING GROQ + HUGGINGFACE EMBEDDINGS
# # -------------------------------

# from dotenv import load_dotenv
# import os

# load_dotenv()  # Load .env file

# # Debug check
# if os.getenv("GROQ_API_KEY"):
#     print("Groq API Key loaded:", os.getenv("GROQ_API_KEY")[:8], "********")
# else:
#     print("⚠️ GROQ_API_KEY not found! Please add it to your .env file.")

# # Imports
# from langchain_groq import ChatGroq
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain.chains import RetrievalQA
# from langchain.schema import Document

# # -------------------------------
# # Step 1 — Sample text data
# # -------------------------------
# text_data = """
# RAG stands for Retrieval-Augmented Generation.
# It helps large language models use external knowledge to answer questions more accurately.
# Instead of relying only on their training data, RAG retrieves relevant documents from a knowledge base.
# These documents are added to the LLM prompt, improving the final response quality.
# LangChain provides tools to build RAG pipelines easily using vector databases and retrievers.
# """

# # -------------------------------
# # Step 2 — Split text into chunks
# # -------------------------------
# text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
# texts = text_splitter.split_text(text_data)
# docs = [Document(page_content=t) for t in texts]

# # -------------------------------
# # Step 3 — Free HuggingFace embeddings
# # -------------------------------
# print("🔹 Creating embeddings using HuggingFace model...")
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# # Store embeddings in FAISS
# db = FAISS.from_documents(docs, embeddings)

# # -------------------------------
# # Step 4 — Groq LLM (New Model)
# # -------------------------------
# retriever = db.as_retriever()

# # ✅ Use supported model
# llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=os.getenv("GROQ_API_KEY"))

# qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# # -------------------------------
# # Step 5 — Ask a question
# # -------------------------------
# query = "What is RAG and why is it used?"
# response = qa.invoke({"query": query})  # .run() deprecated → use .invoke()

# print("\nUser Question:", query)
# print("Answer:", response["result"])






# -------------------------------
# 🧠 STEP 2 — RAG SYSTEM WITH YOUR OWN PDF DATA (Everything inside rag_env)
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
# Imports for LangChain & PDF handling
# -------------------------------
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader

# -------------------------------
# Step 1 — Load your PDF safely
# -------------------------------
# Automatically build absolute path to your PDF
pdf_path = Path(__file__).parent / "data" / "DSA_CompleteNotes.pdf"
pdf_path = pdf_path.resolve()

print(f"📄 Loading PDF from: {pdf_path}")

# Check if file exists
if not pdf_path.exists():
    raise FileNotFoundError(f"❌ Could not find PDF file at {pdf_path}")

# Load the PDF
loader = PyPDFLoader(str(pdf_path))
documents = loader.load()

print(f"✅ Loaded {len(documents)} pages from the PDF.")

# -------------------------------
# Step 2 — Split into chunks
# -------------------------------
print("🔹 Splitting text into manageable chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(documents)
print(f"✅ Split into {len(chunks)} text chunks.")

# -------------------------------
# Step 3 — Create embeddings (HuggingFace)
# -------------------------------
print("🧠 Creating embeddings for your text...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Store vectors in FAISS
db = FAISS.from_documents(chunks, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 3})  # retrieve top 3 relevant chunks

# -------------------------------
# Step 4 — Set up Groq LLM (Llama 3.1)
# -------------------------------
# Use the current active Groq model
llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=os.getenv("GROQ_API_KEY"))

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# -------------------------------
# Step 5 — Ask a Question
# -------------------------------
query = input("\n🤖 Ask a question about your PDF: ")
result = qa.invoke({"query": query})

print("\n-----------------------------")
print("🔍 QUESTION:", query)
print("-----------------------------")
print("💬 ANSWER:", result["result"])
print("-----------------------------")

# (Optional) Show which chunks were used for the answer
for i, doc in enumerate(result["source_documents"], start=1):
    print(f"\n📄 Source {i}: {doc.metadata['source']} (Page {doc.metadata.get('page', '?')})")
    print(doc.page_content[:200], "...")
