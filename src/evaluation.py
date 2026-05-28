import re
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score


# -----------------------------
# 🔧 Text Cleaning
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# -----------------------------
# 📊 ROUGE
# -----------------------------
def compute_rouge(reference, prediction):
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

    reference = clean_text(reference)
    prediction = clean_text(prediction)

    score_val = scorer.score(reference, prediction)
    return score_val['rouge1'].fmeasure


# -----------------------------
# 📊 BLEU
# -----------------------------
def compute_bleu(reference, prediction):
    smoothie = SmoothingFunction().method1

    reference = clean_text(reference).split()
    prediction = clean_text(prediction).split()

    return sentence_bleu([reference], prediction, smoothing_function=smoothie)


# -----------------------------
# 📊 BERTScore
# -----------------------------
def compute_bertscore(reference, prediction):
    try:
        P, R, F1 = score([prediction], [reference], lang="en", verbose=False)
        return F1.mean().item()
    except Exception as e:
        print("BERTScore Error:", e)
        return 0

# -----------------------------
# 📊 Accuracy (Improved)
# -----------------------------
def compute_accuracy(reference, prediction):

    reference = clean_text(reference)
    prediction = clean_text(prediction)

    ref_words = set(reference.split())
    pred_words = set(prediction.split())

    overlap = len(ref_words & pred_words)
    total = len(ref_words)

    if total == 0:
        return 0

    score = overlap / total

    # relaxed threshold
    if score >= 0.3:
        return 1

    return 0