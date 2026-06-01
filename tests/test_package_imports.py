"""Package import contract tests."""

from __future__ import annotations

import subprocess
import sys


def test_root_import_leaves_pipeline_exports_lazy() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            ("import microplex_us; print('build_us_microplex' in vars(microplex_us))"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "False"


def test_root_import_does_not_require_torch_or_core_microplex() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib.abc\n"
                "import sys\n"
                "\n"
                "class BlockTorch(importlib.abc.MetaPathFinder):\n"
                "    def find_spec(self, fullname, path=None, target=None):\n"
                "        if fullname == 'torch' or fullname.startswith('torch.'):\n"
                "            raise ModuleNotFoundError(\"No module named 'torch'\")\n"
                "        return None\n"
                "\n"
                "sys.meta_path.insert(0, BlockTorch())\n"
                "import microplex_us\n"
                "print('microplex' in sys.modules)\n"
                "print('TargetSpec' in vars(microplex_us))\n"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.splitlines() == ["False", "False"]


def test_pe_rebuild_checkpoint_import_does_not_require_torch() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib.abc\n"
                "import sys\n"
                "\n"
                "class BlockTorch(importlib.abc.MetaPathFinder):\n"
                "    def find_spec(self, fullname, path=None, target=None):\n"
                "        if fullname == 'torch' or fullname.startswith('torch.'):\n"
                "            raise ModuleNotFoundError(\"No module named 'torch'\")\n"
                "        return None\n"
                "\n"
                "sys.meta_path.insert(0, BlockTorch())\n"
                "import microplex_us.pipelines.pe_us_data_rebuild_checkpoint\n"
                "print('microplex' in sys.modules)\n"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "False"


def test_pe_rebuild_checkpoint_help_does_not_require_torch_or_core_microplex() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib.abc\n"
                "import runpy\n"
                "import sys\n"
                "\n"
                "class BlockTorch(importlib.abc.MetaPathFinder):\n"
                "    def find_spec(self, fullname, path=None, target=None):\n"
                "        if fullname == 'torch' or fullname.startswith('torch.'):\n"
                "            raise ModuleNotFoundError(\"No module named 'torch'\")\n"
                "        return None\n"
                "\n"
                "sys.meta_path.insert(0, BlockTorch())\n"
                "sys.argv = [\n"
                "    'pe_us_data_rebuild_checkpoint',\n"
                "    '--help',\n"
                "]\n"
                "try:\n"
                "    runpy.run_module(\n"
                "        'microplex_us.pipelines.pe_us_data_rebuild_checkpoint',\n"
                "        run_name='__main__',\n"
                "    )\n"
                "except SystemExit as exc:\n"
                "    print(f'exit={exc.code}')\n"
                "print(f'microplex_imported={\"microplex\" in sys.modules}')\n"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.splitlines()[-2:] == [
        "exit=0",
        "microplex_imported=False",
    ]


def test_data_sources_import_leaves_family_benchmark_lazy() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import microplex_us.data_sources; "
                "print('microplex_us.data_sources.family_imputation_benchmark' "
                "in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "False"
