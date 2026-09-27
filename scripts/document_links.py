"""Keep generated Markdown links correct after reports moved under docs/."""

from __future__ import annotations

import posixpath
import re
from pathlib import Path


DOCUMENT_PATHS = {
    "ANALYSIS.md": "docs/archive/ANALYSIS.md",
    "DIRECTION_RESULTS.md": "docs/archive/DIRECTION_RESULTS.md",
    "NEURAL_COMPARISON.md": "docs/archive/NEURAL_COMPARISON.md",
    "NEURAL_DIRECTION.md": "docs/archive/NEURAL_DIRECTION.md",
    "NEURAL_CLASSIFIERS.md": "docs/results/NEURAL_CLASSIFIERS.md",
    "MODERN_DIRECTION_MODELS.md": "docs/results/MODERN_DIRECTION_MODELS.md",
    "IMPROVEMENT_REPORT.md": "docs/results/IMPROVEMENT_REPORT.md",
    "RESULTS_INDEX.md": "docs/results/RESULTS_INDEX.md",
    "RESULTS_GALLERY.md": "docs/results/RESULTS_GALLERY.md",
    "EXTERNAL_CHECK_PROTOCOL.md": "docs/methods/EXTERNAL_CHECK_PROTOCOL.md",
    "ROADMAP.md": "docs/methods/ROADMAP.md",
}
LINK = re.compile(r"(?P<open>!?\[[^\]]*\]\()(?P<target>[^)]+)(?P<close>\))")


def write_report(root: Path, original_name: str, content: str) -> Path:
    """Write Markdown authored as if it lived at the repository root."""
    destination = DOCUMENT_PATHS[original_name]

    def rewrite(match: re.Match[str]) -> str:
        raw = match.group("target")
        path, marker, fragment = raw.partition("#")
        if not path or "://" in path or path.startswith(("mailto:", "data:")):
            return match.group(0)
        target = DOCUMENT_PATHS.get(posixpath.normpath(path), posixpath.normpath(path))
        relative = posixpath.relpath(target, posixpath.dirname(destination))
        return f"{match.group('open')}{relative}{marker}{fragment}{match.group('close')}"

    output = root / destination
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(LINK.sub(rewrite, content))
    return output
