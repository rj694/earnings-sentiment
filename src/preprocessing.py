"""Clean and segment earnings call transcripts for sentiment analysis."""

import logging
import re
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download NLTK data if not already present
for _resource, _path in [
    ("punkt", "tokenizers/punkt"),
    ("punkt_tab", "tokenizers/punkt_tab"),
    ("stopwords", "corpora/stopwords"),
    ("wordnet", "corpora/wordnet"),
]:
    try:
        nltk.data.find(_path)
    except LookupError:
        nltk.download(_resource, quiet=True)

logger = logging.getLogger(__name__)

# Pre-compiled patterns
_SPEAKER_LABEL_RE = re.compile(r"^.{2,80}\s--\s.+$")
_OPERATOR_BRACKET_RE = re.compile(r"\[Operator [^\]]+\]", re.IGNORECASE)
_QA_MARKER_RE = re.compile(r"^Questions (?:&|and) Answers:\s*$")


def remove_boilerplate(text: str) -> str:
    """Strip operator greetings, call metadata, and Motley Fool footer from a transcript."""
    lines = text.split("\n")

    # --- Strip start: everything before the first named speaker ---
    start_idx = 0
    for i, line in enumerate(lines):
        if _SPEAKER_LABEL_RE.match(line.strip()):
            start_idx = i
            break

    # --- Strip end: everything from "Duration:" onward ---
    end_idx = len(lines)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("Duration:"):
            end_idx = i
            break
        if stripped == "Call participants:":
            end_idx = i
            break
    # Fallback: strip Motley Fool footer if Duration/Call participants not found
    if end_idx == len(lines):
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip() == "All earnings call transcripts":
                end_idx = i
                break

    text = "\n".join(lines[start_idx:end_idx])

    # --- Strip in-text operator bracketed tags ---
    text = _OPERATOR_BRACKET_RE.sub("", text)

    # Strip trailing bare "Operator" lines left after bracket removal
    result_lines = text.split("\n")
    while result_lines and result_lines[-1].strip() in ("Operator", ""):
        result_lines.pop()
    text = "\n".join(result_lines)

    # Collapse runs of 3+ newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def segment_transcript(text: str) -> dict[str, str]:
    """Split transcript into prepared remarks and Q&A sections using the marker line."""
    for line in text.split("\n"):
        if _QA_MARKER_RE.match(line.strip()):
            marker = line.strip()
            parts = text.split(marker, 1)
            return {
                "prepared_remarks": parts[0].strip(),
                "qa": parts[1].strip(),
            }
    logger.warning("No Q&A marker found — treating entire transcript as a single section")
    return {"full": text.strip()}


def tokenise_and_clean(text: str) -> list[str]:
    """Lowercase, tokenise, remove stopwords, and lemmatise text."""
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    # NLP note: we keep only alphabetic tokens to drop punctuation and numbers,
    # then remove common English stopwords before lemmatising to base forms.
    return [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token.isalpha() and token not in stop_words
    ]


def process_transcript(path: Path) -> dict:
    """Load a transcript file, clean it, segment it, and return metadata."""
    text = path.read_text(encoding="utf-8")

    # Parse ticker and quarter from filename: AAPL_2021_Q1.txt
    parts = path.stem.split("_")
    ticker = parts[0]
    year = parts[1]
    quarter = parts[2]

    raw_word_count = len(text.split())

    cleaned_text = remove_boilerplate(text)
    cleaned_word_count = len(cleaned_text.split())

    segments = segment_transcript(cleaned_text)
    section_type = "prepared_remarks+qa" if "qa" in segments else "full"

    return {
        "ticker": ticker,
        "year": year,
        "quarter": quarter,
        "raw_word_count": raw_word_count,
        "cleaned_word_count": cleaned_word_count,
        "section_type": section_type,
        "cleaned_text": cleaned_text,
        "segments": segments,
    }
