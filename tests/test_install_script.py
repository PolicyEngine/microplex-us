from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALL_SCRIPT = REPO_ROOT / "scripts/install.sh"


def _run_install(
    *args: str,
    system: str = "Darwin",
    machine: str = "arm64",
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "CONDA_EXE": "conda",
            "MICROPLEX_US_INSTALL_UNAME_S": system,
            "MICROPLEX_US_INSTALL_UNAME_M": machine,
        }
    )
    return subprocess.run(
        ["bash", str(INSTALL_SCRIPT), *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_help_lists_install_modes() -> None:
    result = _run_install("--help")

    assert result.returncode == 0
    assert "--prod" in result.stdout
    assert "--dev" in result.stdout
    assert "--dev-intel-mac" in result.stdout
    assert "--dry-run" in result.stdout


def test_prod_install_rejects_intel_macos() -> None:
    result = _run_install("--prod", "--dry-run", machine="x86_64")

    assert result.returncode == 2
    assert "Production installs on macOS require Apple Silicon" in result.stderr
    assert "./scripts/install.sh --dev-intel-mac" in result.stderr


def test_dev_install_rejects_intel_macos() -> None:
    result = _run_install("--dev", "--dry-run", machine="x86_64")

    assert result.returncode == 2
    assert "Production installs on macOS require Apple Silicon" in result.stderr
    assert "./scripts/install.sh --dev-intel-mac" in result.stderr


def test_prod_install_uses_python_314_on_arm_macos() -> None:
    result = _run_install("--prod", "--dry-run")

    assert result.returncode == 0
    assert "uv sync --python 3.14 --extra policyengine" in result.stdout


def test_dev_install_uses_python_314_on_arm_macos() -> None:
    result = _run_install("--dev", "--dry-run")

    assert result.returncode == 0
    assert (
        "uv sync --python 3.14 --extra dev --extra policyengine"
        in result.stdout
    )


def test_intel_macos_dev_install_uses_conda_forge_environment() -> None:
    result = _run_install("--dev-intel-mac", "--dry-run", machine="x86_64")

    assert result.returncode == 0
    assert "conda env update --file" in result.stdout
    assert "envs/macos-intel-conda-forge.yml --prune" in result.stdout
    assert "--solver" not in result.stdout
    assert "/envs/microplex-us-intel/bin/python -m pip install" in (
        result.stdout
    )
    assert "--upgrade-strategy only-if-needed -e" in result.stdout
    assert "dev" in result.stdout
    assert "policyengine" in result.stdout
    assert "/envs/microplex-us-intel/bin/python -c" in result.stdout
    assert "torch" in result.stdout


def test_intel_macos_dev_install_rejects_non_intel_platforms() -> None:
    result = _run_install("--dev-intel-mac", "--dry-run")

    assert result.returncode == 2
    assert "--dev-intel-mac is only for Intel macOS" in result.stderr
