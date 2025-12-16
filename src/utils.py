from __future__ import annotations

import io
import re
import tarfile
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Tuple
from urllib.error import HTTPError, URLError


@dataclass
class TexFile:
    name: str
    text: str


def fetch_tex_files_from_tar_gz_url(url: str, max_tex_bytes: int = 5_000_000) -> List[TexFile]:
    """
    Streams a .tar.gz from `url`, returns all .tex files as decoded text.
    max_tex_bytes: safety cap per file to avoid huge memory usage.
    """
    tex_files: List[TexFile] = []

    try:
        with urllib.request.urlopen(url) as resp:
            with tarfile.open(fileobj=resp, mode="r:gz") as tar:
                for member in tar:
                    if not member.isfile():
                        continue
                    if not member.name.lower().endswith(".tex"):
                        continue

                    f = tar.extractfile(member)
                    if f is None:
                        continue

                    data = f.read(max_tex_bytes + 1)
                    if len(data) > max_tex_bytes:
                        # skip overly large files
                        continue

                    # Best-effort decoding for arXiv sources
                    text = data.decode("utf-8", errors="replace")
                    tex_files.append(TexFile(member.name, text))

    except HTTPError as e:
        raise RuntimeError(f"HTTP error {e.code}: {e.reason}") from e
    except URLError as e:
        raise RuntimeError(f"URL error: {e.reason}") from e
    except tarfile.TarError as e:
        raise RuntimeError(f"Invalid tar.gz: {e}") from e

    return tex_files


def chunk_tex_files(tex_files: List[TexFile], max_chars: int = 80_000) -> List[str]:
    """
    Simple chunker that concatenates files with delimiters up to max_chars.
    """
    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0

    for tf in tex_files:
        blob = f"\n\n===== FILE: {tf.name} =====\n{tf.text}\n"
        if len(blob) > max_chars:
            # If a single file is huge, cut it (still gives model partial context)
            blob = blob[:max_chars]

        if cur_len + len(blob) > max_chars and cur:
            chunks.append("".join(cur))
            cur = []
            cur_len = 0

        cur.append(blob)
        cur_len += len(blob)

    if cur:
        chunks.append("".join(cur))

    return chunks


EXTRACTION_PROMPT = """You are given LaTeX source from a research paper.

Task: Extract algorithms defined or clearly described in the paper.

Rules:
- Output ONLY:
  1) One or more LaTeX blocks using the algorithm and algpseudocode packages:
     \\begin{algorithm}
     \\caption{...}
     \\begin{algorithmic}[1]
       \\Require ...
       \\Ensure ...
       \\State ...
     \\end{algorithmic}
     \\end{algorithm}
  2) After each algorithm block, list and briefly describe every variable/parameter/constant appearing.
  3) Then write one concise paragraph describing what the algorithm does and its purpose.
- Preserve original logic, variables, and ordering. Do not invent steps or notation.
- If there are no algorithms, output exactly: NO_ALGORITHMS_FOUND

LaTeX source:
{latex_chunk}
"""


def call_llm(prompt: str) -> str:
    """
    Replace this with your LLM call.
    Must return a string response.
    """
    raise NotImplementedError("Wire this to your LLM provider")


def extract_algorithms_from_tex_via_llm(url: str) -> str:
    tex_files = fetch_tex_files_from_tar_gz_url(url)
    if not tex_files:
        return "NO_TEX_FILES_FOUND"

    # Optional: keep only likely “main” files first (helps if context is tight)
    tex_files.sort(key=lambda t: (("main" not in t.name.lower()), len(t.text)))

    chunks = chunk_tex_files(tex_files, max_chars=80_000)

    outputs: List[str] = []
    for ch in chunks:
        prompt = EXTRACTION_PROMPT.format(latex_chunk=ch)
        outputs.append(call_llm(prompt))

    # If you chunked, you often want a second pass to deduplicate/merge.
    merge_prompt = """You are given multiple partial extractions of algorithms from different chunks.
Merge them into a single final answer. Remove duplicates. Keep the same output rules.
If all parts say NO_ALGORITHMS_FOUND, output exactly NO_ALGORITHMS_FOUND.

Parts:
{parts}
"""
    merged = call_llm(merge_prompt.format(parts="\n\n---\n\n".join(outputs)))
    return merged


if __name__ == "__main__":
    url = "https://example.com/source.tar.gz"
    print(extract_algorithms_from_tex_via_llm(url))
