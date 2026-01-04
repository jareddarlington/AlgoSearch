import re


def _strip_comments(tex: str) -> str:
    """Remove LaTeX comments, but preserve escaped percent \%"""
    out_lines = []
    for line in tex.splitlines():
        # Find first unescaped %
        m = re.search(r"(?<!\\)%", line)
        out_lines.append(line[: m.start()] if m else line)
    return "\n".join(out_lines)


def _remove_bibliography_env(tex: str) -> str:
    """Remove bibliography environments (common and uncommon variants)"""
    # Standard thebibliography
    tex = re.sub(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", "", tex, flags=re.DOTALL)
    # Alternative environment names
    tex = re.sub(r"\\begin\{references\}.*?\\end\{references\}", "", tex, flags=re.DOTALL)
    tex = re.sub(r"\\begin\{biblist\}.*?\\end\{biblist\}", "", tex, flags=re.DOTALL)
    tex = re.sub(r"\\begin\{thebibliography\*\}.*?\\end\{thebibliography\*\}", "", tex, flags=re.DOTALL)
    return tex


def _remove_bib_commands(tex: str) -> str:
    """Remove bibliography commands (BibTeX, BibLaTeX, and variants)"""
    # Standard BibTeX commands
    tex = re.sub(r"\\bibliography\{[^}]*\}", "", tex)
    tex = re.sub(r"\\bibliographystyle\{[^}]*\}", "", tex)

    # BibLaTeX commands
    tex = re.sub(r"\\printbibliography(?:\[[^\]]*\])?", "", tex)
    tex = re.sub(r"\\addbibresource(?:\[[^\]]*\])?\{[^}]*\}", "", tex)
    tex = re.sub(r"\\bibliography\[[^\]]*\]\{[^}]*\}", "", tex)

    # Citation-related commands
    tex = re.sub(r"\\nocite\{[^}]*\}", "", tex)
    tex = re.sub(r"\\bibitem(?:\[[^\]]*\])?\{[^}]*\}", "", tex)

    return tex


def _collapse_whitespace(tex: str) -> str:
    """Remove excessive blank lines"""
    tex = re.sub(r"[ \t]+\n", "\n", tex)
    tex = re.sub(r"\n{3,}", "\n\n", tex)
    return tex.strip()


def has_clean_algorithm_env(tex: str) -> bool:
    """Check if text contains algorithm environments"""
    algo_envs = ["algorithm", "algorithm*", "algorithmic", "algorithmic*", "algpseudocode"]
    for env in algo_envs:
        if f"\\begin{{{env}}}" in tex and f"\\end{{{env}}}" in tex:
            return True
    return False


def clean_tex(tex: str) -> str:
    """
    Minimal LaTeX cleaning: removes comments and bibliography.
    Preserves macro definitions and all content.
    """
    if not has_clean_algorithm_env(tex):
        raise Exception('Paper contains no well-defined algorithm environments')

    # Remove comments
    tex = _strip_comments(tex)

    # Remove bibliography sections and commands
    tex = _remove_bibliography_env(tex)
    tex = _remove_bib_commands(tex)

    # Light cleanup of whitespace
    tex = _collapse_whitespace(tex)

    return tex
