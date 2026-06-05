from __future__ import annotations

from pathlib import Path

import tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_project_metadata_does_not_omit_torch_on_macos_x86_64() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    lock_text = (REPO_ROOT / "uv.lock").read_text()

    uv_config = pyproject.get("tool", {}).get("uv", {})

    assert "required-environments" not in uv_config
    assert "override-dependencies" not in uv_config
    assert "platform_machine != 'x86_64' or sys_platform != 'darwin'" not in (
        lock_text
    )


def test_intel_macos_conda_forge_environment_is_declared() -> None:
    env_text = (REPO_ROOT / "envs/macos-intel-conda-forge.yml").read_text()

    assert "name: microplex-us-intel" in env_text
    assert "  - conda-forge" in env_text
    assert "  - nodefaults" in env_text
    assert "  - python=3.13" in env_text
    assert "  - pytorch=2.11.*" in env_text
    assert "  - pip" in env_text
