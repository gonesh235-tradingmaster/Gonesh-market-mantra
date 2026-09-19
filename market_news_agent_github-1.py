"""
MARKET NEWS INTELLIGENCE AGENT — GITHUB SINGLE-FILE VERSION
===========================================================

Upload this ONE file to GitHub.

What it does:
1. Fetches news from Google News RSS (no news API key needed)
2. Removes duplicate articles
3. Sends each segment as a batch to Gemini
4. Returns structured JSON analysis
5. Saves the final report to ./data/
6. Prints a readable report to the console

Gemini API key:
- GitHub Actions: create repository secret GEMINI_API_KEY
- Local machine: set GEMINI_API_KEY as an environment variable

Do NOT put your real API key inside this file.

GitHub repository secret:
Settings -> Secrets and variables -> Actions -> New repository secret
Name: GEMINI_API_KEY
Value: your Gemini API key

Run locally:
    python market_news_agent_github.py

Optional environment variables:
    GEMINI_MODEL=gemini-3.8-flash
    MAX_ARTICLES_PER_SEGMENT=5
    RSS_RESULTS_PER_QUERY=8
"""

# ============================================================
# AUTO-INSTALL REQUIRED PACKAGES
# ============================================================

import importlib.util
import subprocess
import sys


REQUIRED_PACKAGES = {
    "feedparser": "feedparser",
    "pydantic": "pydantic",
    "google": "google-genai",
}


def ensure_dependencies():
    missing = []

    for module_name, package_name in REQUIRED_PACKAGES.items():
        if importlib.util.find_spec(module_name) is None:
            missing.append(package_name)

    if missing:
        print(
            "[SETUP] Installing missing packages:",
            ", ".join(missing)
        )

        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                *missing,
            ]
        )


ensure_dependencies()


# ============================================================
# IMPORTS
# ============================================================

import os
import re
import json
import time
import hashlib
from datetime import datetime, timezone
from urllib.parse import quote
from pathlib import Path
from typing import List, Literal

import feedparser
from pydantic import BaseModel, Field
from google import genai
from google.genai import types


# ============================================================
# CONFIG
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "\nGEMINI_API_KEY is missing.\n\n"
        "GitHub Actions:\n"
        "Settings -> Secrets and variables -> Actions\n"
        "Create secret: GEMINI_API_KEY\n\n"
        "Local:\n"
        "Set GEMINI_API_KEY in your environment.\n"
    )


MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash"
)

MAX_ARTICLES_PER_SEGMENT = int(
    os.getenv(
        "MAX_ARTICLES_PER_SEGMENT",
        "5"
    )
)

RSS_RESULTS_PER_QUERY = int(
    os.getenv(
        "RSS_RESULTS_PER_QUERY",
        "8"
    )
)

RSS_DELAY_SECONDS = float(
    os.getenv(
        "RSS_DELAY_SECONDS",
        "0.4"
    )
)

MAX_RETRIES = 3
RETRY_BASE_SECONDS = 2

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# NEWS SEGMENTS
# ============================================================

SEGMENTS = {

    "Oil & Energy": [
        "crude oil price",
        "OPEC oil news",
    ],

    "Bond & Central Bank": [
        "US Treasury yield",
        "Federal Reserve interest rate",
        "RBI monetary policy",
    ],

    "War & Geopolitics": [
        "geopolitical conflict",
        "war geopolitical news",
    ],

    "Tariff & Trade": [
        "tariff trade war",
        "India US tariff trade",
    ],

    "Gold & Metals": [
        "gold price",
        "silver metals market",
    ],

    "Indian Market": [
        "Nifty Sensex stock market India",
        "Indian stock market news",
    ],

    "Crypto Market": [
        "Bitcoin crypto market",
        "BTC cryptocurrency news",
    ],

    "Global Indices": [
        "Nasdaq S&P 500 global stock market",
        "Nikkei Hang Seng market",
    ],
}


# ============================================================
# GEMINI STRUCTURED OUTPUT
# ============================================================

class ArticleAnalysis(BaseModel):

    summary: str = Field(
        description=(
            "A factual 2-3 sentence summary of the "
            "article using only the supplied information."
        )
    )

    india_market_impact: str = Field(
        description=(
            "Potential impact on Indian equities, "
            "especially Nifty and Sensex, including "
            "the transmission mechanism when relevant."
        )
    )

    crypto_market_impact: str = Field(
        description=(
            "Potential impact on Bitcoin and broader "
            "crypto markets."
        )
    )

    criticality: Literal[
        "Low",
        "Medium",
        "High",
    ]

    sentiment: Literal[
        "Bullish",
        "Bearish",
        "Neutral",
    ]

    confidence: int = Field(
        ge=0,
        le=100,
        description=(
            "Confidence from 0 to 100 in the analysis."
        )
    )

    event_type: Literal[
        "Macro",
        "Central Bank",
        "Inflation",
        "Rates",
        "Bond",
        "Oil",
        "Gold",
        "Metals",
        "Geopolitics",
        "War",
        "Tariff",
        "Trade",
        "Equity",
        "Crypto",
        "Corporate",
        "Other",
    ]

    time_horizon: Literal[
        "Immediate",
        "1D",
        "1W",
        "Longer",
    ]


class BatchArticleResult(BaseModel):

    index: int = Field(
        description="Original article index."
    )

    analysis: ArticleAnalysis


class BatchAnalysis(BaseModel):

    articles: List[BatchArticleResult]


# ============================================================
# HELPERS
# ============================================================

def clean_html(text):
    if not text:
        return ""

    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def normalize_title(title):
    title = (
        title
        .lower()
        .strip()
    )

    title = re.sub(
        r"[^a-z0-9\s]",
        " ",
        title
    )

    title = re.sub(
        r"\s+",
        " ",
        title
    )

    return title.strip()


def article_hash(article):

    raw = (
        normalize_title(
            article.get("title", "")
        )
        + "|"
        + article.get("link", "")
    )

    return hashlib.md5(
        raw.encode("utf-8")
    ).hexdigest()


# ============================================================
# GOOGLE NEWS RSS
# ============================================================

def fetch_google_news_rss(
    query,
    max_results=8
):

    url = (
        "https://news.google.com/rss/search?"
        f"q={quote(query)}"
        "&hl=en-IN"
        "&gl=IN"
        "&ceid=IN:en"
    )

    try:
        feed = feedparser.parse(url)

    except Exception as exc:
        print(
            f"[RSS ERROR] {query}: {exc}"
        )
        return []

    articles = []

    for entry in feed.entries[:max_results]:

        title = entry.get(
            "title",
            ""
        ).strip()

        if not title:
            continue

        source_name = "Unknown"

        try:
            source_name = (
                entry.source.get(
                    "title",
                    "Unknown"
                )
            )
        except Exception:
            pass

        articles.append({

            "title": title,

            "link": entry.get(
                "link",
                ""
            ),

            "published": entry.get(
                "published",
                ""
            ),

            "source": source_name,

            "raw_summary": clean_html(
                entry.get(
                    "summary",
                    ""
                )
            ),
        })

    return articles


# ============================================================
# COLLECT NEWS
# ============================================================

def collect_all_segments():

    all_news = {}

    for segment, keywords in SEGMENTS.items():

        print(
            f"\n[FETCH] {segment}"
        )

        segment_articles = []

        seen_titles = set()
        seen_hashes = set()

        for keyword in keywords:

            articles = fetch_google_news_rss(
                keyword,
                RSS_RESULTS_PER_QUERY
            )

            for article in articles:

                title_key = normalize_title(
                    article["title"]
                )

                hash_key = article_hash(
                    article
                )

                if title_key in seen_titles:
                    continue

                if hash_key in seen_hashes:
                    continue

                seen_titles.add(
                    title_key
                )

                seen_hashes.add(
                    hash_key
                )

                segment_articles.append(
                    article
                )

                if (
                    len(segment_articles)
                    >= MAX_ARTICLES_PER_SEGMENT
                ):
                    break

            if (
                len(segment_articles)
                >= MAX_ARTICLES_PER_SEGMENT
            ):
                break

            time.sleep(
                RSS_DELAY_SECONDS
            )

        all_news[segment] = (
            segment_articles
        )

        print(
            f"       -> "
            f"{len(segment_articles)} unique articles"
        )

    return all_news


# ============================================================
# GEMINI PROMPT
# ============================================================

def build_prompt(
    segment,
    articles
):

    blocks = []

    for index, article in enumerate(
        articles
    ):

        blocks.append(
            f"""
ARTICLE INDEX: {index}

TITLE:
{article["title"]}

SOURCE:
{article.get("source", "Unknown")}

PUBLISHED:
{article.get("published", "")}

SNIPPET:
{article.get("raw_summary", "")}

--------------------------------
"""
        )

    articles_text = "\n".join(
        blocks
    )

    return f"""
You are a professional macroeconomic and
financial-market news analyst.

SEGMENT:
{segment}

Analyze EVERY supplied article independently.

Rules:

1. Use only the title and snippet supplied.
2. Never invent facts.
3. Keep factual summary separate from interpretation.
4. Assess possible Indian equity impact,
   especially Nifty and Sensex.
5. Assess possible Bitcoin/crypto impact.
6. Do not provide BUY/SELL trading advice.
7. Criticality means market importance.
8. Sentiment means likely directional market tone.
9. Use lower confidence when information is unclear.
10. Return exactly one result for every article.
11. Preserve the original article index.

Criticality definitions:

LOW:
Limited broad-market relevance.

MEDIUM:
Potentially meaningful for a sector, asset,
or short-term market move.

HIGH:
Major macroeconomic, central-bank, geopolitical,
tariff, commodity, or financial-market event.

Sentiment:

BULLISH:
Potentially supportive.

BEARISH:
Potentially negative.

NEUTRAL:
Limited or balanced directional effect.

Time horizon:

Immediate = hours
1D = next trading day
1W = several trading sessions
Longer = multi-week or structural

Articles:

{articles_text}
"""


# ============================================================
# GEMINI ANALYSIS
# ============================================================

def analyze_segment(
    segment,
    articles
):

    if not articles:
        return []

    prompt = build_prompt(
        segment,
        articles
    )

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            response = client.models.generate_content(

                model=MODEL_NAME,

                contents=prompt,

                config=types.GenerateContentConfig(

                    temperature=0.2,

                    response_mime_type=(
                        "application/json"
                    ),

                    response_schema=BatchAnalysis,
                ),
            )

            parsed = getattr(
                response,
                "parsed",
                None
            )

            if parsed is None:

                parsed = (
                    BatchAnalysis.model_validate_json(
                        response.text
                    )
                )

            analyzed_by_index = {
                item.index: item.analysis.model_dump()
                for item in parsed.articles
            }

            results = []

            for index, article in enumerate(
                articles
            ):

                analysis = analyzed_by_index.get(
                    index
                )

                if analysis is None:

                    analysis = {
                        "summary": (
                            article.get(
                                "raw_summary",
                                ""
                            )
                            or article.get(
                                "title",
                                ""
                            )
                        )[:500],

                        "india_market_impact":
                            "Analysis unavailable.",

                        "crypto_market_impact":
                            "Analysis unavailable.",

                        "criticality":
                            "Low",

                        "sentiment":
                            "Neutral",

                        "confidence":
                            0,

                        "event_type":
                            "Other",

                        "time_horizon":
                            "Immediate",
                    }

                results.append({
                    **article,
                    **analysis,
                })

            return results

        except Exception as exc:

            print(
                f"[GEMINI ERROR] "
                f"{segment} | "
                f"attempt {attempt}/{MAX_RETRIES}: "
                f"{exc}"
            )

            if attempt < MAX_RETRIES:

                wait_seconds = (
                    RETRY_BASE_SECONDS
                    * (2 ** (attempt - 1))
                )

                time.sleep(
                    wait_seconds
                )

    return [
        {
            **article,
            "summary": (
                article.get(
                    "raw_summary",
                    ""
                )
                or article.get(
                    "title",
                    ""
                )
            )[:500],

            "india_market_impact":
                "Analysis unavailable.",

            "crypto_market_impact":
                "Analysis unavailable.",

            "criticality":
                "Low",

            "sentiment":
                "Neutral",

            "confidence":
                0,

            "event_type":
                "Other",

            "time_horizon":
                "Immediate",

            "analysis_error":
                "Gemini failed after retries.",
        }
        for article in articles
    ]


# ============================================================
# ANALYZE ALL SEGMENTS
# ============================================================

def analyze_all(
    all_news
):

    final_report = {}

    for segment, articles in all_news.items():

        print(
            f"\n[AI] {segment} "
            f"({len(articles)} articles)"
        )

        final_report[segment] = (
            analyze_segment(
                segment,
                articles
            )
        )

    return final_report


# ============================================================
# SAVE REPORT
# ============================================================

def save_report(
    final_report
):

    now = datetime.now(
        timezone.utc
    )

    timestamp = now.strftime(
        "%Y%m%d_%H%M%S"
    )

    filename = (
        OUTPUT_DIR
        / f"market_news_report_{timestamp}.json"
    )

    output = {

        "agent":
            "Market News Intelligence Agent",

        "phase":
            "Phase 1",

        "generated_at":
            now.isoformat(),

        "model":
            MODEL_NAME,

        "segments":
            final_report,
    }

    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False
        )

    return filename


# ============================================================
# PRINT REPORT
# ============================================================

def print_report(
    final_report
):

    print("\n")
    print("=" * 80)
    print(
        " MARKET NEWS INTELLIGENCE REPORT "
    )
    print("=" * 80)

    total_articles = 0

    for segment, articles in (
        final_report.items()
    ):

        print("\n")
        print("-" * 80)
        print(segment)
        print("-" * 80)

        for article in articles:

            total_articles += 1

            print(
                f"\n[{article['criticality']}] "
                f"[{article['sentiment']}] "
                f"[Confidence: "
                f"{article['confidence']}%]"
            )

            print(
                f"Event: "
                f"{article['event_type']} | "
                f"Horizon: "
                f"{article['time_horizon']}"
            )

            print(
                f"\nHeadline:\n"
                f"{article['title']}"
            )

            print(
                f"\nSummary:\n"
                f"{article['summary']}"
            )

            print(
                f"\nIndia Impact:\n"
                f"{article['india_market_impact']}"
            )

            print(
                f"\nCrypto Impact:\n"
                f"{article['crypto_market_impact']}"
            )

            print(
                f"\nSource: "
                f"{article.get('source', 'Unknown')}"
            )

            print(
                f"Published: "
                f"{article.get('published', '')}"
            )

            print(
                f"URL: "
                f"{article.get('link', '')}"
            )

    print("\n")
    print(
        f"Total articles analyzed: "
        f"{total_articles}"
    )

    print("=" * 80)


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    print("=" * 80)
    print(
        " MARKET NEWS INTELLIGENCE AGENT — PHASE 1"
    )
    print("=" * 80)

    print(
        f"Model: {MODEL_NAME}"
    )

    print(
        "\n[1/4] Collecting Google News RSS..."
    )

    raw_news = (
        collect_all_segments()
    )

    print(
        "\n[2/4] Running Gemini analysis..."
    )

    final_report = (
        analyze_all(
            raw_news
        )
    )

    print(
        "\n[3/4] Saving JSON report..."
    )

    report_file = save_report(
        final_report
    )

    print(
        "\n[4/4] Printing report..."
    )

    print_report(
        final_report
    )

    elapsed = (
        time.time() - start_time
    )

    print(
        f"\n✅ DONE"
        f"\nReport: {report_file}"
        f"\nRuntime: {elapsed:.2f} seconds"
    )


if __name__ == "__main__":
    main()
