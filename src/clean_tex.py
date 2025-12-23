import re
from typing import Iterable, Optional

# Environments to drop entirely (including their contents).
DEFAULT_DROP_ENVS = {
    "figure",
    "figure*",
    "table",
    "table*",
    "wrapfigure",
    "wraptable",
    "sidewaystable",
    "sidewaysfigure",
    "tikzpicture",
    "tabular",
}

# Environments to always keep (algorithm-related + common code blocks).
DEFAULT_ALGO_ENVS = {
    "algorithm",
    "algorithm*",
    "algorithmic",
    "algorithmic*",
    "algpseudocode",
}

DEFAULT_CODE_ENVS = {
    "lstlisting",
    "verbatim",
    "Verbatim",
    "minted",
}

# Commands whose braced argument can be safely removed or replaced.
DEFAULT_DROP_COMMANDS = {
    r"\title",
    r"\author",
    r"\date",
    r"\thanks",
    r"\maketitle",
    r"\hypersetup",
    r"\bibliography",
    r"\bibliographystyle",
}

# Inline formatting commands to unwrap: \textbf{X} -> X (same for others)
DEFAULT_UNWRAP_COMMANDS = {
    r"\textit",
    r"\emph",
    r"\underline",
}


def _strip_comments(tex: str) -> str:
    # Remove LaTeX comments, but preserve escaped percent \%
    # Also try to avoid nuking percent in URLs inside \url{...} (still ok usually).
    out_lines = []
    for line in tex.splitlines():
        # Find first unescaped %
        m = re.search(r"(?<!\\)%", line)
        out_lines.append(line[: m.start()] if m else line)
    return "\n".join(out_lines)


def _remove_environment_blocks(tex: str, env_names: Iterable[str]) -> str:
    # Remove \begin{env} ... \end{env} (non-nested robust enough for most papers)
    for env in env_names:
        pattern = re.compile(
            rf"\\begin\{{{re.escape(env)}\}}.*?\\end\{{{re.escape(env)}\}}",
            re.DOTALL,
        )
        tex = pattern.sub("", tex)
    return tex


def _remove_bibliography_env(tex: str) -> str:
    return re.sub(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", "", tex, flags=re.DOTALL)


def _drop_commands_with_optional_arg(tex: str, cmd: str) -> str:
    # Drops occurrences of \cmd[...]{...} or \cmd{...} or \cmd[...]
    # For \maketitle / no-arg commands, this also matches and removes them.
    # This is intentionally conservative (won't fully parse nested braces).
    cmd_re = re.escape(cmd)
    pattern = re.compile(
        rf"{cmd_re}"
        r"(?:\s*\[[^\]]*\])?"  # optional [...]
        r"(?:\s*\{[^{}]*\})?",  # optional { ... } (no nesting)
        flags=re.DOTALL,
    )
    return pattern.sub("", tex)


def _unwrap_simple_commands(tex: str, cmd: str) -> str:
    # Unwrap simple formatting commands with a single braced arg:
    # \cmd{X} -> X (handles one-level braces only)
    cmd_re = re.escape(cmd)
    pattern = re.compile(rf"{cmd_re}\s*\{{([^{{}}]*)\}}")
    while True:
        new = pattern.sub(r"\1", tex)
        if new == tex:
            return tex
        tex = new


def _normalize_refs_and_cites(tex: str) -> str:
    tex = re.sub(r"\\eqref\{([^}]*)\}", r"EQREF(\1)", tex)
    tex = re.sub(r"\\ref\{([^}]*)\}", r"REF(\1)", tex)
    tex = re.sub(r"\\cite[t|p]?\*?(?:\[[^\]]*\])*\{[^}]*\}", "[CITATION]", tex)
    return tex


def _collapse_whitespace(tex: str) -> str:
    # Remove excessive blank lines
    tex = re.sub(r"[ \t]+\n", "\n", tex)
    tex = re.sub(r"\n{3,}", "\n\n", tex)
    return tex.strip()


def has_clean_algorithm_env(tex: str) -> bool:
    for env in DEFAULT_ALGO_ENVS:
        if f"\\begin{{{env}}}" in tex and f"\\end{{{env}}}" in tex:
            return True
    return False


def clean_tex(
    tex: str,
    *,
    drop_envs: Optional[Iterable[str]] = None,
    keep_envs: Optional[Iterable[str]] = None,
    drop_commands: Optional[Iterable[str]] = None,
    unwrap_commands: Optional[Iterable[str]] = None,
    keep_alg_context_lines: int = 0,
) -> str:
    """
    Prune LaTeX text down to content that preserves algorithm definitions.

    - Drops common non-algorithm environments (figures/tables/tikz) and bibliography.
    - Keeps algorithm/code environments as-is.
    - Removes comments and some metadata commands.
    - Unwraps basic formatting commands.

    If keep_alg_context_lines > 0, it will retain N lines of context before/after
    kept algorithm/code environments, and drop the rest of the document body.
    This is high-recall for "keep only algorithm regions".
    """
    drop_envs = set(drop_envs or DEFAULT_DROP_ENVS)
    keep_envs = set(keep_envs or DEFAULT_ALGO_ENVS)
    drop_commands = list(drop_commands or DEFAULT_DROP_COMMANDS)
    unwrap_commands = list(unwrap_commands or DEFAULT_UNWRAP_COMMANDS)

    if not has_clean_algorithm_env(tex):
        raise Exception('Paper contains no well-defined algorithm environments')

    tex = _strip_comments(tex)
    tex = _remove_bibliography_env(tex)

    # If user wants only algorithm regions + context, do an extract-first approach.
    if keep_alg_context_lines > 0:
        lines = tex.splitlines()
        keep = [False] * len(lines)

        # Mark lines covered by keep_env blocks (+ context window)
        for env in keep_envs:
            for m in re.finditer(
                rf"\\begin\{{{re.escape(env)}\}}.*?\\end\{{{re.escape(env)}\}}",
                tex,
                flags=re.DOTALL,
            ):
                # Convert char spans to line spans
                start_char, end_char = m.span()
                start_line = tex.count("\n", 0, start_char)
                end_line = tex.count("\n", 0, end_char)
                a = max(0, start_line - keep_alg_context_lines)
                b = min(len(lines) - 1, end_line + keep_alg_context_lines)
                for i in range(a, b + 1):
                    keep[i] = True

        # Also keep paragraphs containing common algorithm keywords (prose algorithms)
        algo_kw = re.compile(
            r"\b(Algorithm|pseudocode|procedure|initialize|initialise|repeat|until|for each|while)\b",
            re.IGNORECASE,
        )
        for i, line in enumerate(lines):
            if algo_kw.search(line):
                a = max(0, i - keep_alg_context_lines)
                b = min(len(lines) - 1, i + keep_alg_context_lines)
                for j in range(a, b + 1):
                    keep[j] = True

        tex = "\n".join(line for i, line in enumerate(lines) if keep[i])

    # Drop non-algorithm environments (after any extraction)
    # But do not drop any env that is also in keep_envs.
    tex = _remove_environment_blocks(tex, drop_envs - keep_envs)

    # Drop some metadata commands
    for cmd in drop_commands:
        tex = _drop_commands_with_optional_arg(tex, cmd)

    # Normalize refs/cites
    tex = _normalize_refs_and_cites(tex)

    # Unwrap formatting commands
    for cmd in unwrap_commands:
        tex = _unwrap_simple_commands(tex, cmd)

    # Light cleanup
    tex = _collapse_whitespace(tex)
    return tex
