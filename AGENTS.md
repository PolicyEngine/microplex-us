# AGENTS.md

This repository is the US content package for `microplex`.

## Contract

- Keep the package declarative: YAML specs and JSON manifests only.
- Do not add runtime Python, tests, scripts, notebooks, dashboards, generated
  artifacts, or local environment files to this repository.
- Move execution machinery to `microplex`, donor imputation machinery to
  `microimpute`, and calibration machinery to `microcalibrate`.
- Treat any country-specific imperative logic as a missing generic operator or
  adapter in `microplex`, not as a reason to recreate Python here.

## Validation

Use the generic Microplex content-package check:

```bash
PYTHONPATH=/path/to/microplex/src:src uv run --no-project --python 3.13 \
  --with pydantic --with pyyaml \
  python -m microplex.content_package \
  --package microplex_us \
  --spec specs/us-2024.yaml \
  --contract manifests/ecps_export_contract.json \
  --src-root src/microplex_us
```

The check must confirm:

- the spec loads
- `spec.variables` exactly covers the frozen export contract plus declared
  imputation variables
- `src/microplex_us` contains no Python files

Also run:

```bash
find . -name '*.py' -print
uv build
```

The first command should print nothing. The built wheel should contain only the
US spec, JSON manifests, and wheel metadata.
