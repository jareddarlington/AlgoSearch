# python ./src/extraction.py --input_dir ./data/temp --papers_db ./data/papers.db --algorithms_db ./data/algorithms.db --model gemini-2.5-flash

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from db_utils import get_db_connection
from schemas import AlgorithmList
from gemini import create_model, generate_content

PROMPT_PATH = "prompts/extraction.txt"
PROMPT_TEMPLATE = open(PROMPT_PATH, encoding="utf-8").read()


def init_database(db_path: str):
    """Initialize the SQLite database with algorithms table."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS algorithms (
                algo_id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                latex TEXT,
                categories TEXT
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
            print(f"Warning: Failed reading {p}: {e}")
    return result


def try_salvage_partial_json(text: str) -> Optional[Dict[str, Any]]:
    """Attempt to salvage complete algorithms from truncated JSON.

    When the response is truncated due to MAX_TOKENS, we might have several
    complete algorithm objects followed by an incomplete one. This function
    finds the last complete algorithm and cuts off everything after it.
    """
    try:
        if '"algorithms"' not in text:
            return None

        array_start = text.find('"algorithms"')
        if array_start == -1:
            return None

        bracket_start = text.find('[', array_start)
        if bracket_start == -1:
            return None

        # Find all complete algorithm objects by finding "}," patterns
        complete_objects_end = bracket_start
        pos = bracket_start
        while True:
            next_pos = text.find('},', pos + 1)
            if next_pos == -1:
                break
            complete_objects_end = next_pos + 1
            pos = next_pos

        # If we found at least one complete object
        if complete_objects_end > bracket_start:
            truncated = text[:complete_objects_end].rstrip(',').rstrip()
            reconstructed = truncated + ']}'

            try:
                data = json.loads(reconstructed)
                if isinstance(data, dict) and 'algorithms' in data:
                    algos = data.get('algorithms', [])
                    if isinstance(algos, list) and len(algos) > 0:
                        print(f"Salvaged {len(algos)} complete algorithms from truncated JSON")
                        return data
            except json.JSONDecodeError:
                pass

    except Exception:
        pass

    return None


def process_paper(
    client,
    config,
    model_name: str,
    paper_id: str,
    latex: str,
    papers_db_path: str,
    algorithms_db_path: str,
) -> int:
    """Process a single paper and extract algorithms."""
    prompt = PROMPT_TEMPLATE.replace("{latex}", latex)

    try:
        text = generate_content(client, model_name, config, prompt)

        # Try to parse JSON with error handling
        data = None
        is_partial = False
        try:
            data = json.loads(text.strip())
        except json.JSONDecodeError as e:
            # Try to salvage partial results
            data = try_salvage_partial_json(text)
            if data is None:
                raise ValueError(f"Invalid JSON from model: {e.msg}")
            is_partial = True

        parsed = AlgorithmList.model_validate(data)

        # Insert algorithms into database
        algo_count = 0
        with get_db_connection(algorithms_db_path) as conn:
            cursor = conn.cursor()
            for algo in parsed.algorithms:
                cursor.execute(
                    '''
                    INSERT OR REPLACE INTO algorithms
                    (paper_id, name, description, latex, categories)
                    VALUES (?, ?, ?, ?, ?)
                ''',
                    (
                        paper_id,
                        algo.name,
                        algo.description,
                        algo.latex,
                        json.dumps(algo.categories) if algo.categories else None,
                    ),
                )
                algo_count += 1

        error_msg = f"Partial: {algo_count} algorithms" if is_partial else None
        mark_paper_processed(papers_db_path, paper_id, True, error_msg, algo_count, model_name)

        symbol = "⚠" if is_partial else "✓"
        print(f"{symbol} {paper_id}: {algo_count} algorithm(s)")
        return algo_count

    except Exception as e:
        mark_paper_processed(papers_db_path, paper_id, False, str(e), 0, model_name)
        print(f"✗ {paper_id}: {e}")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing .txt files")
    parser.add_argument("--papers_db", type=str, default="data/papers.db", help="Path to papers database")
    parser.add_argument("--algorithms_db", type=str, default="data/algorithms.db", help="Path to algorithms database")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash-lite", help="Gemini model name")
    parser.add_argument("--max_papers", type=int, default=None, help="Max number of papers to process")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input dir not found: {input_dir}")
        return 2

    papers_db_path = Path(args.papers_db)
    if not papers_db_path.exists():
        print(f"Error: Papers database not found: {papers_db_path}")
        return 2

    algorithms_db_path = Path(args.algorithms_db)
    algorithms_db_path.parent.mkdir(parents=True, exist_ok=True)
    init_database(str(algorithms_db_path))

    unprocessed_paper_ids = get_unprocessed_papers(str(papers_db_path))
    if not unprocessed_paper_ids:
        print("No unprocessed papers found")
        return 0

    if args.max_papers:
        unprocessed_paper_ids = unprocessed_paper_ids[: args.max_papers]

    print(f"Found {len(unprocessed_paper_ids)} unprocessed papers")

    latex_files = read_latex_files(input_dir, set(unprocessed_paper_ids))
    print(f"Loaded {len(latex_files)} LaTeX files")

    schema = AlgorithmList.model_json_schema()
    client, config = create_model(
        model_name=args.model,
        temperature=0.0,
        max_output_tokens=8192,
        response_schema=schema,
    )

    total_algorithms = 0
    for i, paper_id in enumerate(unprocessed_paper_ids, start=1):
        if paper_id not in latex_files:
            print(f"Warning: LaTeX file not found for {paper_id}")
            mark_paper_processed(str(papers_db_path), paper_id, False, "LaTeX file not found", 0, args.model)
            continue

        print(f"[{i}/{len(unprocessed_paper_ids)}] Processing {paper_id}")
        algo_count = process_paper(
            client, config, args.model, paper_id, latex_files[paper_id], str(papers_db_path), str(algorithms_db_path)
        )
        total_algorithms += algo_count

    print(f"\nExtraction complete! Total algorithms: {total_algorithms}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
