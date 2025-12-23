#!/usr/bin/env python3
"""
gemini_batch_algo_extract.py

Batch-extract algorithm records from LaTeX sources using the Gemini Batch API
and structured JSON output (JSON Schema from Pydantic).

Requires:
  pip install requests pydantic

Auth:
  export GEMINI_API_KEY="..."

Usage examples:
  # Submit a batch and poll until finished, writing JSONL outputs:
  python gemini_batch_algo_extract.py \
    --input_dir ./tex_sources \
    --out_jsonl ./algorithms.jsonl \
    --out_papers_dir ./paper_outputs \
    --batch_display_name "algosearch-extract-001"

Notes:
- Uses the Gemini Developer API Batch endpoint:
    POST https://generativelanguage.googleapis.com/v1beta/models/{model}:batchGenerateContent
- Polls with:
    GET  https://generativelanguage.googleapis.com/v1beta/{name=batches/*}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import requests

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from structure import AlgorithmList


# -----------------------------
# Prompt
# -----------------------------

PROMPT_TEMPLATE = """You are extracting algorithm records from a LaTeX research paper.

Task:
- Identify every clearly-defined algorithm in the LaTeX source.
- Prefer algorithms in environments like: algorithm, algorithmic, algpseudocode, procedure, or well-structured step lists.
- If there are no algorithms, return an empty list.

For each algorithm, output:
- name (+ aliases if present)
- short_description (3-6 sentences; problem solved, core idea, when used)
- algorithm_text (normalized: plain text steps, keep variable symbols, remove LaTeX noise)
- inputs, outputs (lists of strings)
- variables: list of {{symbol, role, type?}}
- problem_type, data_structures, paradigm
- time_complexity, space_complexity (if known; else null)
- assumptions, guarantees

Important constraints:
- Output must be valid JSON matching the provided schema.
- Do not include any extra keys.
- If something is unknown, use null (or empty list where appropriate).
- Do not quote the entire paper; extract only algorithm-relevant content.

LaTeX source:
<<<BEGIN_LATEX
{latex}
END_LATEX>>>
"""


# -----------------------------
# Gemini Batch API helpers
# -----------------------------

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"


@dataclass
class GeminiClient:
    api_key: str

    def _headers(self) -> Dict[str, str]:
        return {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }

    def batch_generate_content(
        self, model: str, display_name: str, inlined_requests: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        POST /models/{model}:batchGenerateContent
        Body schema (simplified):
        {
          "displayName": "...",
          "inputConfig": {
            "requests": {
              "requests": [
                {"request": {...GenerateContentRequest...}, "metadata": {...}},
                ...
              ]
            }
          }
        }
        """
        url = f"{GEMINI_API_BASE}/models/{model}:batchGenerateContent"
        body = {"displayName": display_name, "inputConfig": {"requests": {"requests": inlined_requests}}}
        resp = requests.post(url, headers=self._headers(), data=json.dumps(body), timeout=120)
        resp.raise_for_status()
        return resp.json()

    def get_batch(self, batch_name: str) -> Dict[str, Any]:
        """
        GET /{name=batches/*}
        """
        url = f"{GEMINI_API_BASE}/{batch_name}"
        resp = requests.get(url, headers=self._headers(), timeout=60)
        resp.raise_for_status()
        return resp.json()


# -----------------------------
# IO + batching
# -----------------------------


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def read_tex_files(input_dir: Path, max_files: Optional[int]) -> List[Tuple[Path, str]]:
    paths = sorted([p for p in input_dir.rglob("*.tex") if p.is_file()])
    if max_files is not None:
        paths = paths[:max_files]
    out: List[Tuple[Path, str]] = []
    for p in paths:
        try:
            out.append((p, p.read_text(encoding="utf-8", errors="ignore")))
        except Exception as e:
            print(f"[warn] failed reading {p}: {e}", file=sys.stderr)
    return out


def chunked(xs: List[Any], n: int) -> List[List[Any]]:
    return [xs[i : i + n] for i in range(0, len(xs), n)]


def build_generate_content_request(
    prompt: str,
    response_schema: Dict[str, Any],
    temperature: float = 0.0,
    max_output_tokens: int = 8192,
) -> Dict[str, Any]:
    """
    GenerateContentRequest (REST) shape aligns with docs:
      {
        "contents": [{"parts": [{"text": "..."}]}],
        "generationConfig": {
          "temperature": ...,
          "maxOutputTokens": ...,
          "responseMimeType": "application/json",
          "responseJsonSchema": {...}
        }
      }
    """
    return {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
            "responseMimeType": "application/json",
            "responseJsonSchema": response_schema,
        },
    }


def safe_json_loads(s: str) -> Any:
    # Gemini sometimes returns JSON with leading/trailing whitespace.
    s2 = s.strip()
    return json.loads(s2)


def extract_response_text(gen_content_response: Dict[str, Any]) -> str:
    """
    GenerateContentResponse JSON can contain candidates[0].content.parts[0].text.
    Use a conservative extractor.
    """
    candidates = gen_content_response.get("candidates") or []
    if not candidates:
        raise ValueError("No candidates in response")
    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    texts = []
    for part in parts:
        if isinstance(part, dict) and "text" in part:
            texts.append(part["text"])
    if not texts:
        # Some responses expose a top-level "text" in client SDK; REST typically uses parts.
        raise ValueError("No text parts found in response")
    return "".join(texts)


# -----------------------------
# Main
# -----------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, required=True, help="Directory containing .tex files (recursive).")
    ap.add_argument("--out_jsonl", type=str, required=True, help="Output JSONL: one line per algorithm record.")
    ap.add_argument("--out_papers_dir", type=str, default="", help="Optional: write per-paper JSON to this directory.")
    ap.add_argument("--model", type=str, default="gemini-2.5-flash-lite", help="Gemini model name.")
    ap.add_argument("--batch_display_name", type=str, default="algosearch-batch", help="Batch display name.")
    ap.add_argument("--max_files", type=int, default=None, help="Max number of .tex files to process.")
    ap.add_argument("--requests_per_batch", type=int, default=50, help="How many papers per batch job.")
    ap.add_argument("--poll_interval_s", type=float, default=10.0, help="Polling interval.")
    ap.add_argument("--max_poll_s", type=float, default=6 * 60 * 60, help="Max time to poll before giving up.")
    args = ap.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("Missing GEMINI_API_KEY env var.", file=sys.stderr)
        return 2

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Input dir not found: {input_dir}", file=sys.stderr)
        return 2

    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    out_papers_dir = Path(args.out_papers_dir) if args.out_papers_dir else None
    if out_papers_dir:
        out_papers_dir.mkdir(parents=True, exist_ok=True)

    # Prepare schema for structured output
    schema = AlgorithmList.model_json_schema()

    # Load inputs
    tex_files = read_tex_files(input_dir, args.max_files)
    if not tex_files:
        print("No .tex files found.", file=sys.stderr)
        return 1

    # Build batch requests
    client = GeminiClient(api_key=api_key)
    batches = chunked(tex_files, args.requests_per_batch)

    # Open JSONL for append
    f_jsonl = out_jsonl.open("a", encoding="utf-8")

    try:
        for batch_idx, group in enumerate(batches, start=1):
            display_name = f"{args.batch_display_name}-{batch_idx:04d}"
            inlined_requests: List[Dict[str, Any]] = []

            for path, latex in group:
                paper_id = sha256_text(str(path.resolve()))
                latex_hash = sha256_text(latex)

                prompt = PROMPT_TEMPLATE.format(latex=latex)

                gen_req = build_generate_content_request(
                    prompt=prompt,
                    response_schema=schema,
                    temperature=0.0,
                    max_output_tokens=8192,
                )

                inlined_requests.append(
                    {
                        "request": gen_req,
                        "metadata": {
                            "paper_id": paper_id,
                            "source_path": str(path),
                            "latex_sha256": latex_hash,
                        },
                    }
                )

            # Submit the batch
            submit = client.batch_generate_content(
                model=args.model,
                display_name=display_name,
                inlined_requests=inlined_requests,
            )

            # The response is an Operation; the batch resource name is typically in "name"
            # and can be polled via batches.get. We’ll attempt to find a batches/* name.
            op_name = submit.get("name", "")
            if not op_name:
                raise RuntimeError(f"Unexpected submit response (no name): {submit}")

            # Sometimes the operation name can be "batches/..." already; sometimes it is an operation
            # that includes the batch in metadata. We handle the common case: op_name starts with batches/.
            batch_name = op_name if op_name.startswith("batches/") else op_name

            # Poll until state is SUCCEEDED / FAILED / CANCELLED
            start = time.time()
            while True:
                if time.time() - start > args.max_poll_s:
                    raise TimeoutError(f"Polling exceeded max time for {batch_name}")

                status = client.get_batch(batch_name)

                state = status.get("state", "")
                # Known states include: BATCH_STATE_UNSPECIFIED, RUNNING, SUCCEEDED, FAILED, CANCELLED
                if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
                    break

                time.sleep(args.poll_interval_s)

            # Read outputs (inlinedResponses for inlined input)
            output = status.get("output") or {}
            inlined = (output.get("inlinedResponses") or {}).get("inlinedResponses") or []

            if not inlined:
                # If responses are written to a file, you would see output.responsesFile
                # This script uses inlined requests; if you hit size limits, switch to file-based input.
                responses_file = output.get("responsesFile")
                raise RuntimeError(f"No inlinedResponses found. responsesFile={responses_file!r}")

            # Process each response in order; write per-algorithm JSONL lines
            for item in inlined:
                meta = item.get("metadata") or {}
                paper_id = meta.get("paper_id")
                source_path = meta.get("source_path")
                latex_sha256 = meta.get("latex_sha256")

                if "error" in item:
                    err = item.get("error") or {}
                    rec = {
                        "paper_id": paper_id,
                        "source_path": source_path,
                        "latex_sha256": latex_sha256,
                        "error": err,
                    }
                    f_jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    continue

                resp = item.get("response") or {}
                text = extract_response_text(resp)

                data = safe_json_loads(text)
                parsed = AlgorithmList.model_validate(data)

                # Optional: save one JSON per paper
                if out_papers_dir:
                    per_paper = {
                        "paper_id": paper_id,
                        "source_path": source_path,
                        "latex_sha256": latex_sha256,
                        "algorithms": [a.model_dump() for a in parsed.algorithms],
                    }
                    (out_papers_dir / f"{paper_id}.json").write_text(
                        json.dumps(per_paper, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )

                # JSONL: one line per algorithm (easier for indexing)
                for algo in parsed.algorithms:
                    rec = algo.model_dump()
                    rec["_paper_id"] = paper_id
                    rec["_source_path"] = source_path
                    rec["_latex_sha256"] = latex_sha256
                    f_jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")

            f_jsonl.flush()

    finally:
        f_jsonl.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
