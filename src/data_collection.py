"""Load Kaggle earnings call transcripts and download stock prices via yfinance."""

import logging
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# --- Configuration ---

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KAGGLE_PICKLE = PROJECT_ROOT / "data" / "raw" / "kaggle_source" / "motley-fool-data.pkl"
TRANSCRIPTS_DIR = PROJECT_ROOT / "data" / "raw" / "transcripts"
PRICES_DIR = PROJECT_ROOT / "data" / "raw" / "prices"

# Candidate companies — we'll drop any with fewer than MIN_TRANSCRIPTS
TARGET_TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "JPM", "JNJ", "BAC"]
MIN_QUARTER = "2021-Q1"
MAX_QUARTER = "2023-Q2"
MIN_TRANSCRIPTS = 7

# yfinance date range — pad a few months around our transcript window
PRICE_START = "2020-12-01"
PRICE_END = "2023-09-30"


def load_raw_data(pickle_path: Path) -> pd.DataFrame:
    """Load the raw Motley Fool pickle file into a DataFrame."""
    logger.info("Loading raw data from %s", pickle_path)
    return pd.read_pickle(pickle_path)


def filter_by_tickers_and_dates(
    df: pd.DataFrame, tickers: list[str], min_q: str, max_q: str
) -> pd.DataFrame:
    """Filter DataFrame to target tickers and quarter range (inclusive)."""
    mask_ticker = df["ticker"].isin(tickers)
    mask_quarter = (df["q"] >= min_q) & (df["q"] <= max_q)
    filtered = df[mask_ticker & mask_quarter].copy()
    logger.info(
        "Filtered to %d rows for %d tickers in %s–%s",
        len(filtered), len(tickers), min_q, max_q,
    )
    return filtered


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the longest transcript for each ticker/quarter pair."""
    df["transcript_len"] = df["transcript"].str.len()
    deduped = (
        df.sort_values("transcript_len", ascending=False)
        .drop_duplicates(subset=["ticker", "q"], keep="first")
        .drop(columns=["transcript_len"])
    )
    dropped = len(df) - len(deduped)
    if dropped:
        logger.info("Dropped %d duplicate rows (kept longest per ticker/quarter)", dropped)
    return deduped


def enforce_min_transcripts(
    df: pd.DataFrame, min_count: int
) -> tuple[pd.DataFrame, list[str]]:
    """Drop companies with fewer than min_count transcripts; return filtered df and dropped tickers."""
    counts = df.groupby("ticker").size()
    keep = counts[counts >= min_count].index.tolist()
    drop = counts[counts < min_count].index.tolist()
    for ticker in drop:
        logger.warning(
            "Dropping %s — only %d transcripts (minimum is %d)",
            ticker, counts[ticker], min_count,
        )
    return df[df["ticker"].isin(keep)].copy(), drop


def save_transcripts(df: pd.DataFrame, output_dir: Path) -> int:
    """Save each transcript as {TICKER}_{YEAR}_Q{QUARTER}.txt; return count saved."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for _, row in df.iterrows():
        # q format is "2022-Q1" → year=2022, quarter=1
        year, quarter_str = row["q"].split("-")
        quarter = quarter_str  # already "Q1", "Q2", etc.
        filename = f"{row['ticker']}_{year}_{quarter}.txt"
        filepath = output_dir / filename
        filepath.write_text(row["transcript"], encoding="utf-8")
        saved += 1
    logger.info("Saved %d transcript files to %s", saved, output_dir)
    return saved


def download_prices(
    tickers: list[str], start: str, end: str, output_dir: Path
) -> tuple[int, list[str]]:
    """Download adjusted close prices per ticker from yfinance; return (success_count, failed_tickers)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    success = 0
    failed = []

    for i, ticker in enumerate(tickers):
        if i > 0:
            time.sleep(2)  # avoid yfinance rate limiting

        logger.info("Downloading prices for %s (%d/%d)", ticker, i + 1, len(tickers))
        try:
            price_df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
            if price_df.empty:
                raise ValueError(f"Empty result for {ticker}")
            # yf.download returns MultiIndex columns when single ticker; flatten
            if isinstance(price_df.columns, pd.MultiIndex):
                price_df.columns = price_df.columns.get_level_values(0)
            filepath = output_dir / f"{ticker}.csv"
            price_df.to_csv(filepath)
            logger.info("Saved %d price rows for %s", len(price_df), ticker)
            success += 1
        except Exception as e:
            logger.warning("Failed to download %s: %s — retrying in 30s", ticker, e)
            time.sleep(30)
            try:
                price_df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
                if price_df.empty:
                    raise ValueError(f"Empty result for {ticker} on retry")
                if isinstance(price_df.columns, pd.MultiIndex):
                    price_df.columns = price_df.columns.get_level_values(0)
                filepath = output_dir / f"{ticker}.csv"
                price_df.to_csv(filepath)
                logger.info("Saved %d price rows for %s (retry succeeded)", len(price_df), ticker)
                success += 1
            except Exception as e2:
                logger.error("Permanently failed to download %s: %s — skipping", ticker, e2)
                failed.append(ticker)

    return success, failed


def print_summary(
    transcript_df: pd.DataFrame,
    dropped_tickers: list[str],
    price_successes: int,
    price_failures: list[str],
) -> None:
    """Print a final summary of what was collected."""
    print("\n" + "=" * 60)
    print("DATA COLLECTION SUMMARY")
    print("=" * 60)

    print(f"\nTranscripts saved: {len(transcript_df)}")
    counts = transcript_df.groupby("ticker").size().sort_index()
    for ticker, count in counts.items():
        quarters = sorted(transcript_df[transcript_df["ticker"] == ticker]["q"].unique())
        print(f"  {ticker}: {count} transcripts  ({quarters[0]} – {quarters[-1]})")

    if dropped_tickers:
        print(f"\nCompanies dropped (< {MIN_TRANSCRIPTS} transcripts): {dropped_tickers}")

    print(f"\nPrice files downloaded: {price_successes}/{price_successes + len(price_failures)}")
    if price_failures:
        print(f"  Failed: {price_failures}")

    print("=" * 60 + "\n")


def main() -> None:
    """Run the full data collection pipeline."""
    raw_df = load_raw_data(KAGGLE_PICKLE)
    filtered = filter_by_tickers_and_dates(raw_df, TARGET_TICKERS, MIN_QUARTER, MAX_QUARTER)
    deduped = deduplicate(filtered)
    final_df, dropped = enforce_min_transcripts(deduped, MIN_TRANSCRIPTS)

    final_tickers = sorted(final_df["ticker"].unique().tolist())
    logger.info("Final company list: %s", final_tickers)

    save_transcripts(final_df, TRANSCRIPTS_DIR)
    price_ok, price_fail = download_prices(final_tickers, PRICE_START, PRICE_END, PRICES_DIR)

    print_summary(final_df, dropped, price_ok, price_fail)


if __name__ == "__main__":
    main()
