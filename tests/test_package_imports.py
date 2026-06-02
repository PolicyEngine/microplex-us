"""Package import contract tests."""

from __future__ import annotations

import subprocess
import sys
import textwrap


def _run_python(source: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        check=True,
        capture_output=True,
        text=True,
    )


_BLOCK_TORCH_IMPORTS = """
import importlib.abc
import sys


class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ModuleNotFoundError("No module named 'torch'")
        return None


sys.meta_path.insert(0, BlockTorch())
"""


def test_root_import_leaves_pipeline_exports_lazy() -> None:
    result = _run_python(
        "import microplex_us; print('build_us_microplex' in vars(microplex_us))"
    )

    assert result.stdout.strip() == "False"


def test_root_import_does_not_require_torch_or_core_microplex() -> None:
    result = _run_python(
        _BLOCK_TORCH_IMPORTS
        + """
import microplex_us
print("microplex" in sys.modules)
print("TargetSpec" in vars(microplex_us))
        """
    )

    assert result.stdout.splitlines() == ["False", "False"]


def test_pe_rebuild_checkpoint_import_does_not_require_torch() -> None:
    result = _run_python(
        _BLOCK_TORCH_IMPORTS
        + """
import microplex_us.pipelines.pe_us_data_rebuild_checkpoint
print("microplex" in sys.modules)
        """
    )

    assert result.stdout.strip() == "False"


def test_pe_rebuild_checkpoint_help_does_not_require_torch_or_core_microplex() -> None:
    result = _run_python(
        _BLOCK_TORCH_IMPORTS
        + """
import runpy

sys.argv = ["pe_us_data_rebuild_checkpoint", "--help"]
try:
    runpy.run_module(
        "microplex_us.pipelines.pe_us_data_rebuild_checkpoint",
        run_name="__main__",
    )
except SystemExit as exc:
    print(f"exit={exc.code}")
print(f"microplex_imported={'microplex' in sys.modules}")
        """
    )

    assert result.stdout.splitlines()[-2:] == [
        "exit=0",
        "microplex_imported=False",
    ]


def test_data_sources_import_leaves_family_benchmark_lazy() -> None:
    result = _run_python(
        """
        import sys
        import microplex_us.data_sources
        print("microplex_us.data_sources.family_imputation_benchmark" in sys.modules)
        """
    )

    assert result.stdout.strip() == "False"
