"""Score earnings call transcripts using Loughran-McDonald dictionary and FinBERT."""

import logging
from pathlib import Path

import nltk
import pandas as pd
import torch
from nltk.tokenize import sent_tokenize
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Ensure NLTK sentence tokenizer is available
for _resource, _path in [
    ("punkt", "tokenizers/punkt"),
    ("punkt_tab", "tokenizers/punkt_tab"),
]:
    try:
        nltk.data.find(_path)
    except LookupError:
        nltk.download(_resource, quiet=True)

logger = logging.getLogger(__name__)

# Categories from the LM dictionary that we score
LM_CATEGORIES = ("Positive", "Negative", "Uncertainty", "Litigious", "Constraining")


def load_lm_dictionary(path: Path) -> dict[str, set[str]]:
    """Load the Loughran-McDonald CSV and return sets of lowercase words per sentiment category."""
    df = pd.read_csv(path)
    return {
        cat: set(df.loc[df[cat] > 0, "Word"].str.lower())
        for cat in LM_CATEGORIES
    }


def score_lm(tokens: list[str], lm_dict: dict[str, set[str]]) -> dict[str, float]:
    """Calculate proportion of tokens in each LM category plus net sentiment."""
    total = len(tokens)
    if total == 0:
        return {cat.lower(): 0.0 for cat in LM_CATEGORIES} | {"net_sentiment": 0.0}

    scores = {}
    for cat, words in lm_dict.items():
        matched = sum(1 for t in tokens if t in words)
        scores[cat.lower()] = matched / total

    scores["net_sentiment"] = scores["positive"] - scores["negative"]
    return scores


def load_finbert() -> tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    """Load ProsusAI/finbert tokenizer and model, set to eval mode."""
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    model.eval()
    return tokenizer, model


def score_finbert_sentences(
    sentences: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    batch_size: int = 16,
) -> dict[str, float]:
    """Score sentences in batches with FinBERT and return mean probabilities per label."""
    if not sentences:
        return {"positive": 0.0, "negative": 0.0, "neutral": 0.0, "net_sentiment": 0.0}

    all_probs = []

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        # NLP note: truncation=True ensures sentences longer than 512 tokens
        # are cut rather than causing an error — FinBERT's positional embeddings
        # only go to 512.
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        all_probs.append(probs)

    combined = torch.cat(all_probs, dim=0)
    mean_probs = combined.mean(dim=0)

    # NLP note: FinBERT's label mapping is {0: positive, 1: negative, 2: neutral}.
    # We take the mean probability across all sentences as the transcript-level score.
    label_map = model.config.id2label
    scores = {label_map[i]: mean_probs[i].item() for i in range(len(mean_probs))}
    scores["net_sentiment"] = scores["positive"] - scores["negative"]
    return scores


def score_finbert_transcript(
    text: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    batch_size: int = 16,
) -> dict[str, float]:
    """Split transcript into sentences, score with FinBERT, and return aggregated probabilities."""
    # NLP note: sent_tokenize uses Punkt, a pre-trained unsupervised sentence
    # boundary detector. It handles abbreviations and decimal numbers well,
    # which matters for earnings calls full of "$1.2 billion" and "Mr. Cook".
    sentences = sent_tokenize(text)
    return score_finbert_sentences(sentences, tokenizer, model, batch_size)
