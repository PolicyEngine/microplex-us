#!/usr/bin/env bash
set -euo pipefail

PYTHON_VERSION="${MICROPLEX_US_PYTHON_VERSION:-3.14}"
INTEL_ENV_NAME="microplex-us-intel"

usage() {
  cat <<'USAGE'
Usage: ./scripts/install.sh [--prod|--dev|--dev-intel-mac] [--dry-run]

Install modes:
  --prod            Install the production PolicyEngine runtime with uv.
  --dev             Install development and PolicyEngine dependencies with uv.
  --dev-intel-mac   Install the Intel macOS development environment via conda-forge.

Options:
  --dry-run         Print commands instead of running them.
  --help            Show this help.

Production macOS installs require Apple Silicon (arm64). Intel macOS is
development/testing-only; use --dev-intel-mac there.
USAGE
}

repo_root() {
  cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
}

detect_uname_s() {
  if [[ -n "${MICROPLEX_US_INSTALL_UNAME_S:-}" ]]; then
    printf "%s\n" "$MICROPLEX_US_INSTALL_UNAME_S"
  else
    uname -s
  fi
}

detect_uname_m() {
  if [[ -n "${MICROPLEX_US_INSTALL_UNAME_M:-}" ]]; then
    printf "%s\n" "$MICROPLEX_US_INSTALL_UNAME_M"
  else
    uname -m
  fi
}

run_cmd() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf "+"
    printf " %q" "$@"
    printf "\n"
  else
    "$@"
  fi
}

resolve_intel_env_python() {
  local conda_path
  local conda_root
  if [[ "$CONDA_EXE" == */* ]]; then
    conda_path="$CONDA_EXE"
  elif command -v "$CONDA_EXE" >/dev/null 2>&1; then
    conda_path="$(command -v "$CONDA_EXE")"
  elif [[ "$DRY_RUN" == "1" ]]; then
    printf "<%s-python>\n" "$INTEL_ENV_NAME"
    return
  else
    conda_path="$(command -v "$CONDA_EXE")"
  fi
  conda_root="$(cd "$(dirname "$conda_path")/.." && pwd)"
  printf "%s/envs/%s/bin/python\n" "$conda_root" "$INTEL_ENV_NAME"
}

intel_mac_message() {
  cat <<'MESSAGE' >&2
Production installs on macOS require Apple Silicon (arm64).
This Intel Mac path is development-only; use ./scripts/install.sh --dev-intel-mac.
MESSAGE
}

require_intel_mac() {
  if [[ "$UNAME_S" != "Darwin" || "$UNAME_M" != "x86_64" ]]; then
    cat <<'MESSAGE' >&2
--dev-intel-mac is only for Intel macOS development/testing.
Use ./scripts/install.sh --dev on Apple Silicon macOS and Linux.
MESSAGE
    exit 2
  fi
}

reject_intel_mac_runtime() {
  if [[ "$UNAME_S" == "Darwin" && "$UNAME_M" == "x86_64" ]]; then
    intel_mac_message
    exit 2
  fi
}

MODE="prod"
DRY_RUN="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prod)
      MODE="prod"
      ;;
    --dev)
      MODE="dev"
      ;;
    --dev-intel-mac)
      MODE="dev-intel-mac"
      ;;
    --dry-run)
      DRY_RUN="1"
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      printf "Unknown option: %s\n\n" "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

REPO_ROOT="$(repo_root)"
ENV_FILE="$REPO_ROOT/envs/macos-intel-conda-forge.yml"
CONDA_EXE="${CONDA_EXE:-conda}"
UNAME_S="$(detect_uname_s)"
UNAME_M="$(detect_uname_m)"

cd "$REPO_ROOT"

case "$MODE" in
  prod)
    reject_intel_mac_runtime
    run_cmd uv sync --python "$PYTHON_VERSION" --extra policyengine
    ;;
  dev)
    reject_intel_mac_runtime
    run_cmd uv sync --python "$PYTHON_VERSION" --extra dev --extra policyengine
    ;;
  dev-intel-mac)
    require_intel_mac
    run_cmd "$CONDA_EXE" env update --file "$ENV_FILE" --prune
    INTEL_ENV_PYTHON="$(resolve_intel_env_python)"
    run_cmd "$INTEL_ENV_PYTHON" -m pip install \
      --upgrade-strategy only-if-needed -e ".[dev,policyengine]"
    run_cmd "$INTEL_ENV_PYTHON" -c \
      "import platform, torch; print(f'microplex-us Intel dev env ready: {platform.machine()} torch {torch.__version__}')"
    ;;
esac
