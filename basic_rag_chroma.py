# # -------------------------------
# # 🧠 STEP 3 — Persistent RAG System with ChromaDB
# # -------------------------------

# from dotenv import load_dotenv
# import os
# from pathlib import Path

# # Load environment variables
# load_dotenv()

# if os.getenv("GROQ_API_KEY"):
#     print("Groq API Key loaded:", os.getenv("GROQ_API_KEY")[:8], "********")
# else:
#     print("⚠️ GROQ_API_KEY not found! Please add it to your .env file.")

# # -------------------------------
# # LangChain imports
# # -------------------------------
# from langchain_groq import ChatGroq
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
# from langchain.chains import RetrievalQA

# # -------------------------------
# # Step 1 — Load PDFs from data folder
# # -------------------------------
# data_folder = Path(__file__).parent / "data"
# data_folder = data_folder.resolve()

# print(f"📂 Loading all PDFs from: {data_folder}")

# loader = DirectoryLoader(str(data_folder), glob="**/*.pdf", loader_cls=PyPDFLoader)
# documents = loader.load()
# print(f"✅ Loaded {len(documents)} pages from all PDFs.")

# # -------------------------------
# # Step 2 — Split into text chunks
# # -------------------------------
# print("🔹 Splitting text into manageable chunks...")
# splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
# chunks = splitter.split_documents(documents)
# print(f"✅ Split into {len(chunks)} text chunks.")

# # -------------------------------
# # Step 3 — Create or load ChromaDB vector store
# # -------------------------------
# print("🧠 Setting up persistent ChromaDB...")

# chroma_dir = Path(__file__).parent / "chroma_db"
# chroma_dir.mkdir(exist_ok=True)

# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# # Create (or reuse existing) vector database
# db = Chroma.from_documents(
#     documents=chunks,
#     embedding=embeddings,
#     persist_directory=str(chroma_dir)
# )

# db.persist()  # saves database to disk
# print(f"💾 ChromaDB created at: {chroma_dir}")

# # Create retriever
# retriever = db.as_retriever(search_kwargs={"k": 3})

# # -------------------------------
# # Step 4 — Connect to Groq LLM
# # -------------------------------
# llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=os.getenv("GROQ_API_KEY"))

# qa = RetrievalQA.from_chain_type(
#     llm=llm,
#     retriever=retriever,
#     return_source_documents=True
# )

# # -------------------------------
# # Step 5 — Ask Questions
# # -------------------------------
# while True:
#     query = input("\n🤖 Ask a question (or type 'exit' to quit): ")
#     if query.lower() in ["exit", "quit"]:
#         print("👋 Goodbye!")
#         break

#     result = qa.invoke({"query": query})

#     print("\n-----------------------------")
#     print("🔍 QUESTION:", query)
#     print("-----------------------------")
#     print("💬 ANSWER:", result['result'])
#     print("-----------------------------")

#     # Show sources
#     for i, doc in enumerate(result["source_documents"], start=1):
#         print(f"\n📄 Source {i}: {doc.metadata['source']} (Page {doc.metadata.get('page', '?')})")
#         print(doc.page_content[:200], "...")









# -------------------------------
# 🧠 STEP 4.5 — Conversational RAG System with Memory + ChromaDB
# -------------------------------

from dotenv import load_dotenv
import os
from pathlib import Path

# Load environment variables
load_dotenv()

if os.getenv("GROQ_API_KEY"):
    print("Groq API Key loaded:", os.getenv("GROQ_API_KEY")[:8], "********")
else:
    print("⚠️ GROQ_API_KEY not found! Please add it to your .env file.")

# -------------------------------
# Imports
# -------------------------------
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

# 🔹 NEW imports for conversational memory
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# -------------------------------
# Step 1 — Load PDFs from data folder
# -------------------------------
data_folder = Path(__file__).parent / "data"
data_folder = data_folder.resolve()

print(f"📂 Loading all PDFs from: {data_folder}")

loader = DirectoryLoader(str(data_folder), glob="**/*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()
print(f"✅ Loaded {len(documents)} pages from all PDFs.")

# -------------------------------
# Step 2 — Split into text chunks
# -------------------------------
print("🔹 Splitting text into manageable chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(documents)
print(f"✅ Split into {len(chunks)} text chunks.")

# -------------------------------
# Step 3 — Setup persistent ChromaDB
# -------------------------------
chroma_dir = Path(__file__).parent / "chroma_db"
chroma_dir.mkdir(exist_ok=True)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("🧠 Loading or creating ChromaDB...")
db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=str(chroma_dir)
)
db.persist()
print(f"💾 ChromaDB ready at: {chroma_dir}")

retriever = db.as_retriever(search_kwargs={"k": 3})

# -------------------------------
# Step 4 — LLM + Conversational Memory Setup
# -------------------------------
llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=os.getenv("GROQ_API_KEY"))

# 🧠 Create conversational memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 🔗 Conversational Retrieval Chain combines memory + retriever + LLM
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)

# -------------------------------
# Step 5 — Multi-turn Conversation
# -------------------------------
print("\n🤖 Conversational RAG System Ready! Ask me anything (type 'exit' to quit).")

while True:
    query = input("\n🧑 You: ")
    if query.lower() in ["exit", "quit"]:
        print("👋 Goodbye! Chat history saved in memory (in runtime only).")
        break

    result = qa.invoke({"question": query})

    print("\n💬 Assistant:", result["answer"])
