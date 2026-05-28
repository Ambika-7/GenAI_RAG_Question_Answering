import sys
sys.path.append("src")

import pandas as pd

from rag_pipeline import build_pipeline, generate_answer
from evaluation import compute_rouge, compute_bleu, compute_bertscore, compute_accuracy


# -----------------------------
# ⚙️ EXPERIMENT CONFIG
# -----------------------------

# config = {
#     "name": "Experiment 1",
#     "pdf_path": "data/documents/healthcare.pdf",
#     "chunk_size": 500,
#     "embedding_model": "all-MiniLM-L6-v2",
#     "db_type": "faiss",
#     "llm_model": "llama3"
# }


config = {
    "name": "Experiment 2",
    "pdf_path": "data/documents/healthcare.pdf",
    "chunk_size": 300,
    "embedding_model": "BAAI/bge-small-en",
    "db_type": "chroma",
    "llm_model": "gemma"
}



# -----------------------------
# 🚀 Build RAG System
# -----------------------------

print(f"\n🔬 Running {config['name']}...\n")

vector_db = build_pipeline(
    pdf_path=config["pdf_path"],
    chunk_size=config["chunk_size"],
    embedding_model=config["embedding_model"],
    db_type=config["db_type"]
)


# -----------------------------
# 📊 Load Dataset
# -----------------------------

df = pd.read_csv("data/qa_pairs/test_data.csv")

rouge_total = 0
bleu_total = 0
bert_total = 0
accuracy_total = 0

total = len(df)


# -----------------------------
# 🔄 Evaluation Loop
# -----------------------------

for index, row in df.iterrows():

    question = row["Question"]
    ground_truth = row["Ground Truth"]

    print("\n==============================")
    print("Question:", question)

    try:
        prediction, _ = generate_answer(
            vector_db,
            question,
            llm_model=config["llm_model"]
        )
    except Exception as e:
        print("LLM ERROR:", e)
        prediction = ""

    print("Prediction:", prediction)
    print("Ground Truth:", ground_truth)

    rouge = compute_rouge(ground_truth, prediction)
    bleu = compute_bleu(ground_truth, prediction)
    bert = compute_bertscore(ground_truth, prediction)
    acc = compute_accuracy(ground_truth, prediction)

    print(f"ROUGE: {rouge:.2f}, BLEU: {bleu:.2f}, BERT: {bert:.2f}, ACC: {acc}")

    rouge_total += rouge
    bleu_total += bleu
    bert_total += bert
    accuracy_total += acc


# -----------------------------
# 📊 Final Results
# -----------------------------

print("\n===== FINAL RESULTS =====")
print("Experiment:", config["name"])
print("ROUGE:", round(rouge_total / total, 3))
print("BERTScore:", round(bert_total / total, 3))
print("Accuracy:", round(accuracy_total / total, 3))