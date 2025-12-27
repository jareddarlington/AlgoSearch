#!/usr/bin/env python3
"""
extractor.py

Extract algorithm records from LaTeX sources using the Gemini API
with structured JSON output.

Usage:
python ./src/extractor.py --input_dir ./data/temp --papers_db ./data/papers.db --algorithms_db ./data/algorithms.db --model gemini-2.5-flash
"""

import argparse
import hashlib
import json
import logging
import os
import sqlite3
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
from model_output_structure import AlgorithmList

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
PROMPT_PATH = SCRIPT_DIR / "prompts" / "extraction.txt"

try:
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        PROMPT_TEMPLATE = f.read()
except FileNotFoundError:
    logger.error(f"Prompt template not found at {PROMPT_PATH}")
    sys.exit(1)

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2  # seconds
RETRY_DELAY_MAX = 30  # seconds


def sha256_text(s: str) -> str:
    """Compute SHA256 hash of text string."""
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


@contextmanager
def get_db_connection(db_path: str):
    """Context manager for database connections."""
    conn = sqlite3.connect(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_database(db_path: str):
    """Initialize the SQLite database with algorithms table."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS algorithms (
                algo_id TEXT PRIMARY KEY,
                paper_id TEXT NOT NULL,
                hash TEXT NOT NULL,
                name TEXT NOT NULL,
                aliases TEXT,
                description TEXT,
                latex TEXT,
                categories TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        '''
        )
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_paper_id ON algorithms(paper_id)')


def get_unprocessed_papers(papers_db_path: str) -> List[str]:
    """Get list of paper IDs that have processed=False (0) and status='success'."""
    with get_db_connection(papers_db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM papers WHERE processed = 0 AND status = 'success'")
        return [row[0] for row in cursor.fetchall()]


def mark_paper_processed(
    papers_db_path: str,
    paper_id: str,
    success: bool = True,
    extraction_error: Optional[str] = None,
    algo_count: int = 0,
    model: Optional[str] = None,
):
    """Mark a paper as processed in the papers database."""
    with get_db_connection(papers_db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            'UPDATE papers SET processed = ?, extraction_error = ?, algo_count = ?, model = ? WHERE id = ?',
            (1 if success else 0, extraction_error, algo_count, model, paper_id),
        )


def read_latex_files(input_dir: Path, filter_ids: Set[str]) -> Dict[str, str]:
    """Read LaTeX files for specified paper IDs."""
    paths = [p for p in input_dir.rglob("*.txt") if p.is_file() and p.stem in filter_ids]

    result = {}
    for p in paths:
        try:
            result[p.stem] = p.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            logger.warning(f"Failed reading {p}: {e}")
    return result


def generate_content(api_key: str, model: str, prompt: str, response_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Call Gemini API to generate content with structured output, with retry logic."""
    url = f"{GEMINI_API_BASE}/models/{model}:generateContent"
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}

    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 8192,
            "responseMimeType": "application/json",
            "responseJsonSchema": response_schema,
        },
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
        ],
    }

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(url, headers=headers, json=body, timeout=120)
            response.raise_for_status()
            result = response.json()

            # Log raw response for debugging
            logger.debug(f"Raw API response: {json.dumps(result, indent=2)[:500]}...")

            return result

        except requests.exceptions.Timeout as e:
            last_error = f"Request timeout: {e}"
            logger.warning(f"Attempt {attempt + 1}/{MAX_RETRIES} failed: {last_error}")

        except requests.exceptions.RequestException as e:
            last_error = f"Request error: {e}"
            if hasattr(e.response, 'text'):
                logger.warning(f"API Error Response: {e.response.text[:500]}")
            logger.warning(f"Attempt {attempt + 1}/{MAX_RETRIES} failed: {last_error}")

        except Exception as e:
            last_error = f"Unexpected error: {e}"
            logger.warning(f"Attempt {attempt + 1}/{MAX_RETRIES} failed: {last_error}")

        # Exponential backoff with jitter
        if attempt < MAX_RETRIES - 1:
            delay = min(RETRY_DELAY_BASE * (2**attempt), RETRY_DELAY_MAX)
            logger.info(f"Retrying in {delay} seconds...")
            time.sleep(delay)

    raise Exception(f"Failed after {MAX_RETRIES} attempts. Last error: {last_error}")


def extract_response_text(response: Dict[str, Any]) -> str:
    """Extract text from Gemini API response with detailed error handling."""
    # Check for blocked/filtered content
    if "promptFeedback" in response:
        feedback = response["promptFeedback"]
        if feedback.get("blockReason"):
            raise ValueError(
                f"Content blocked: {feedback.get('blockReason')}. Safety ratings: {feedback.get('safetyRatings')}"
            )

    candidates = response.get("candidates", [])
    if not candidates:
        logger.error(f"No candidates in response. Full response: {json.dumps(response, indent=2)[:1000]}")
        raise ValueError("No candidates in response")

    candidate = candidates[0]

    # Check finish reason
    finish_reason = candidate.get("finishReason")
    if finish_reason and finish_reason not in ["STOP", "MAX_TOKENS"]:
        logger.warning(f"Unusual finish reason: {finish_reason}. Safety ratings: {candidate.get('safetyRatings')}")

    if finish_reason == "MAX_TOKENS":
        logger.warning("Response truncated due to MAX_TOKENS - may result in malformed JSON")

    content = candidate.get("content", {})
    parts = content.get("parts", [])

    texts = []
    for part in parts:
        if isinstance(part, dict) and "text" in part:
            texts.append(part["text"])

    if not texts:
        logger.error(f"No text parts found. Candidate: {json.dumps(candidate, indent=2)[:1000]}")
        raise ValueError(f"No text parts found in response. Finish reason: {finish_reason}")

    return "".join(texts)


def process_paper(
    api_key: str,
    model: str,
    paper_id: str,
    latex: str,
    schema: Dict[str, Any],
    papers_db_path: str,
    algorithms_db_path: str,
) -> int:
    """Process a single paper and extract algorithms."""
    hash_val = sha256_text(latex)
    prompt = PROMPT_TEMPLATE.replace("{latex}", latex)

    try:
        response = generate_content(api_key, model, prompt, schema)
        text = extract_response_text(response)

        # Log first 500 chars of extracted text for debugging
        logger.debug(f"Extracted text preview: {text[:500]}...")

        # Try to parse JSON with better error handling
        try:
            data = json.loads(text.strip())
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error at position {e.pos}: {e.msg}")
            logger.error(f"Problematic JSON snippet: ...{text[max(0, e.pos-100):e.pos+100]}...")
            raise ValueError(f"Invalid JSON from model: {e.msg}")

        parsed = AlgorithmList.model_validate(data)

        # Insert algorithms into database
        algo_count = 0
        with get_db_connection(algorithms_db_path) as conn:
            cursor = conn.cursor()
            for idx, algo in enumerate(parsed.algorithms, start=1):
                algo_id = f"{paper_id}#{idx}"
                cursor.execute(
                    '''
                    INSERT OR REPLACE INTO algorithms
                    (algo_id, paper_id, hash, name, aliases, description, latex, categories)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''',
                    (
                        algo_id,
                        paper_id,
                        hash_val,
                        algo.name,
                        json.dumps(algo.aliases) if algo.aliases else None,
                        algo.description,
                        algo.latex,
                        json.dumps(algo.categories) if algo.categories else None,
                    ),
                )
                algo_count += 1

        mark_paper_processed(papers_db_path, paper_id, success=True, algo_count=algo_count, model=model)
        logger.info(f"✓ {paper_id}: {algo_count} algorithm(s) extracted")
        return algo_count

    except Exception as e:
        error_msg = str(e)
        logger.error(f"✗ {paper_id}: {error_msg}")
        mark_paper_processed(
            papers_db_path, paper_id, success=False, extraction_error=error_msg, algo_count=0, model=model
        )
        return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing .txt files")
    parser.add_argument("--papers_db", type=str, default="data/papers.db", help="Path to papers database")
    parser.add_argument("--algorithms_db", type=str, default="data/algorithms.db", help="Path to algorithms database")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash-lite", help="Gemini model name")
    parser.add_argument("--max_papers", type=int, default=None, help="Max number of papers to process")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        logger.error("Missing GEMINI_API_KEY env var")
        return 2

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Input dir not found: {input_dir}")
        return 2

    papers_db_path = Path(args.papers_db)
    if not papers_db_path.exists():
        logger.error(f"Papers database not found: {papers_db_path}")
        return 2

    algorithms_db_path = Path(args.algorithms_db)
    algorithms_db_path.parent.mkdir(parents=True, exist_ok=True)
    init_database(str(algorithms_db_path))

    # Get unprocessed papers
    unprocessed_paper_ids = get_unprocessed_papers(str(papers_db_path))
    if not unprocessed_paper_ids:
        logger.info("No unprocessed papers found")
        return 0

    if args.max_papers:
        unprocessed_paper_ids = unprocessed_paper_ids[: args.max_papers]

    logger.info(f"Found {len(unprocessed_paper_ids)} unprocessed papers")

    # Load LaTeX files
    latex_files = read_latex_files(input_dir, set(unprocessed_paper_ids))
    logger.info(f"Loaded {len(latex_files)} LaTeX files")

    # Prepare schema
    schema = AlgorithmList.model_json_schema()

    # Process each paper
    total_algorithms = 0
    for i, paper_id in enumerate(unprocessed_paper_ids, start=1):
        if paper_id not in latex_files:
            logger.warning(f"LaTeX file not found for {paper_id}")
            mark_paper_processed(
                str(papers_db_path),
                paper_id,
                success=False,
                extraction_error="LaTeX file not found",
                algo_count=0,
                model=args.model,
            )
            continue

        logger.info(f"[{i}/{len(unprocessed_paper_ids)}] Processing {paper_id}")
        algo_count = process_paper(
            api_key, args.model, paper_id, latex_files[paper_id], schema, str(papers_db_path), str(algorithms_db_path)
        )
        total_algorithms += algo_count

    logger.info(f"\nExtraction complete!")
    logger.info(f"Total algorithms extracted: {total_algorithms}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
