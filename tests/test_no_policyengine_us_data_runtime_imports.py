"""Runtime dependency boundaries for the MP package."""

from __future__ import annotations

import ast
from pathlib import Path


def test_microplex_package_has_no_policyengine_us_data_imports():
    repo_root = Path(__file__).resolve().parents[1]
    package_root = repo_root / "src" / "microplex_us"
    offenders: list[str] = []
    for path in sorted(package_root.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "policyengine_us_data"
            ):
                offenders.append(f"{path.relative_to(repo_root)}:{node.lineno}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("policyengine_us_data"):
                        offenders.append(f"{path.relative_to(repo_root)}:{node.lineno}")

    assert offenders == []
