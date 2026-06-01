from __future__ import annotations

from pathlib import Path

import tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_uv_lock_requires_macos_x86_64_environment() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())

    uv_config = pyproject["tool"]["uv"]

    assert "sys_platform == 'darwin' and platform_machine == 'x86_64'" in (
        uv_config["required-environments"]
    )
    assert "torch; sys_platform != 'darwin' or platform_machine != 'x86_64'" in (
        uv_config["override-dependencies"]
    )
