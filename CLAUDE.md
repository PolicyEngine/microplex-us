# CLAUDE.md

`microplex-us` is a declarative content package.

- Keep package contents to specs and manifests.
- Do not add Python implementation code here.
- Put generic execution, validation, imputation, and calibration machinery in
  `microplex`, `microimpute`, or `microcalibrate`.
- Validate this repository with `microplex.content_package` and a no-`.py` file
  scan before publishing changes.
