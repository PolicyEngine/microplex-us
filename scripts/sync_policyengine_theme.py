"""Sync exported PolicyEngine design tokens into browser-readable CSS variables."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def default_source(repo_root: Path) -> Path:
    """Return the first local PolicyEngine token export next to this repo."""

    candidates = (
        repo_root.parent / "policyengine.org" / "packages" / "config" / "theme.css",
        repo_root.parent / "policyengine" / "packages" / "config" / "theme.css",
        repo_root.parent / "policyengine" / "apps" / "web" / "src" / "app" / "globals.css",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find exported PolicyEngine theme. Searched: {searched}")


def render_browser_tokens(
    source_text: str,
    *,
    source_path: Path,
    repo_root: Path,
) -> str:
    """Convert a Tailwind v4 @theme block into CSS custom properties."""

    match = re.search(r"@theme\s*\{(?P<body>.*?)\}", source_text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No @theme block found in {source_path}")
    body = match.group("body").strip()
    try:
        display_source = source_path.relative_to(repo_root.parent)
    except ValueError:
        display_source = source_path
    return (
        "/* Generated from the exported PolicyEngine design tokens.\n"
        f"   Source: {display_source}\n"
        "   Re-run: python scripts/sync_policyengine_theme.py\n"
        "*/\n"
        ":root {\n"
        f"{body}\n"
        "}\n"
    )


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Sync @policyengine/config theme tokens into dashboard CSS."
    )
    parser.add_argument("--source", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "dashboard" / "policyengine-theme.css",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    source = (args.source or default_source(repo_root)).expanduser().resolve()
    rendered = render_browser_tokens(
        source.read_text(),
        source_path=source,
        repo_root=repo_root,
    )

    if args.check:
        current = args.output.read_text() if args.output.exists() else ""
        if current != rendered:
            print(f"{args.output} is not synced with {source}", file=sys.stderr)
            return 1
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
