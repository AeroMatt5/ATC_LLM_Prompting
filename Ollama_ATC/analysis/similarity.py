"""
Core similarity metrics for comparing ATC transcript text.

Supports TF-IDF cosine similarity, BLEU, ROUGE-L, and optional
sentence-embedding cosine similarity via sentence-transformers.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


def compute_tfidf_cosine(reference: str, hypothesis: str) -> float:
    """TF-IDF weighted cosine similarity between two strings."""
    if not reference.strip() or not hypothesis.strip():
        return 0.0
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([reference, hypothesis])
    return float(sklearn_cosine(tfidf[0:1], tfidf[1:2])[0][0])


def compute_bleu(reference: str, hypothesis: str) -> float:
    """Smoothed BLEU score treating each string as a bag of tokens."""
    if not reference.strip() or not hypothesis.strip():
        return 0.0
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    smooth = SmoothingFunction().method1
    return float(sentence_bleu([ref_tokens], hyp_tokens,
                               smoothing_function=smooth))


def compute_rouge_l(reference: str, hypothesis: str) -> float:
    """ROUGE-L F1 score (longest common subsequence overlap)."""
    if not reference.strip() or not hypothesis.strip():
        return 0.0
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return float(scores["rougeL"].fmeasure)


def compute_semantic_cosine(reference: str, hypothesis: str, model) -> float:
    """Cosine similarity between sentence-transformer embeddings.

    Returns None when model is None so callers can detect unavailability.
    """
    if model is None:
        return None
    if not reference.strip() or not hypothesis.strip():
        return 0.0
    emb = model.encode([reference, hypothesis])
    dot = np.dot(emb[0], emb[1])
    denom = np.linalg.norm(emb[0]) * np.linalg.norm(emb[1])
    if denom == 0:
        return 0.0
    return float(dot / denom)


def compute_all_metrics(reference: str, hypothesis: str,
                        model=None) -> dict:
    """Return a dict of all available similarity metrics for one pair."""
    metrics = {
        "tfidf_cosine": compute_tfidf_cosine(reference, hypothesis),
        "bleu": compute_bleu(reference, hypothesis),
        "rouge_l": compute_rouge_l(reference, hypothesis),
    }
    sem = compute_semantic_cosine(reference, hypothesis, model)
    if sem is not None:
        metrics["semantic_cosine"] = sem
    return metrics


# Ordered list used by plotting functions to iterate deterministically.
METRIC_NAMES = ["tfidf_cosine", "bleu", "rouge_l", "semantic_cosine"]
