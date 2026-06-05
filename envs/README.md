# Developer Testing Environments

Use the normal `uv` install path unless PyPI cannot provide the needed binary
packages for your platform. This folder holds development-only environments for
those cases.

## Intel macOS

Production macOS installs require Apple Silicon (`arm64`). Intel macOS
(`x86_64`) is supported only for development and testing through conda-forge,
because modern PyPI `torch` wheels are not available for that platform.

Create or update the Intel Mac development environment with:

```bash
./scripts/install.sh --dev-intel-mac
```

That command uses `envs/macos-intel-conda-forge.yml` to install Python 3.13 and
PyTorch 2.11 from conda-forge, then installs this repository with the `dev` and
`policyengine` extras using pip inside the conda environment.

Use the normal install script on Apple Silicon macOS and Linux:

```bash
./scripts/install.sh --prod
./scripts/install.sh --dev
```
