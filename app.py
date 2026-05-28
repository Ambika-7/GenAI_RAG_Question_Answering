import sys
sys.path.append("src")

from rag_pipeline import build_pipeline, generate_answer


# -----------------------------
# ⚙️ CONFIG (change for experiments)
# -----------------------------

PDF_PATH = "data/documents/healthcare.pdf"

CHUNK_SIZE = 500
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DB_TYPE = "faiss"
LLM_MODEL = "llama3"


# -----------------------------
# 🚀 Build System
# -----------------------------

print("🔄 Building RAG system...")

vector_db = build_pipeline(
    pdf_path=PDF_PATH,
    chunk_size=CHUNK_SIZE,
    embedding_model=EMBEDDING_MODEL,
    db_type=DB_TYPE
)

print("✅ System Ready!\n")


# -----------------------------
# 💬 Interactive QA Loop
# -----------------------------

while True:
    query = input("Ask a question (type 'exit' to quit): ")

    if query.lower() == "exit":
        print("👋 Exiting...")
        break

    answer, docs = generate_answer(
        vector_db,
        query,
        llm_model=LLM_MODEL
    )

    print("\n🧠 Answer:")
    print(answer)

    print("\n📚 Retrieved Context:")
    for i, doc in enumerate(docs):
        print(f"\n--- Chunk {i+1} ---")
        print(doc.page_content[:300])  # limit text

    print("\n" + "="*50 + "\n")