# Arch → Ledger: full product rename — staged migration plan

Scope (confirmed): rename the **Arch** product to **Ledger** across all repos +
artifact dirs + the R2 bucket. **Whole-word, identifier-aware only** — never a
substring replace (`search`, `architecture`, `hierarchical`, `march` all contain
"arch").

## Product surface (what "arch" actually is)
- **GitHub repo**: `PolicyEngine/arch-data` → `PolicyEngine/ledger-data`. The repo
  now lives in the **PolicyEngine** org (moved from Cosilico). The **local clone
  also moves** `~/CosilicoAI/arch` → `~/PolicyEngine/ledger-data` (mirror org/repo)
  as part of the rename — a single move (org + name) done during the coordination
  window, NOT now (the tree has 17 uncommitted Codex changes; moving a live/dirty
  dir risks lost work). `git remote` already points at `PolicyEngine/arch-data`.
- **Python package**: `cosilico-arch` (modules `arch`, `db`, `micro`, `calibration`) → `cosilico-ledger` (module `arch` → `ledger`)
- **R2 bucket**: `arch` (production CPS/ACS/SOI microdata + calibration targets) → `ledger`
- **microplex-us**: `src/microplex_us/targets/arch.py` (+ console scripts `*-arch-target-*`), references in `us.py`, `performance.py`, `pe_us_data_rebuild_checkpoint.py`, `pe_us_recalibrate_from_checkpoint.py`; **49 `artifacts/arch_*` dirs**; ~1141 "arch" lines (many are substring — exclude).
- **cosilico.ai / docs**: prose + diagrams referencing "Arch".

## Hard blocker — coordinate with Codex first
Both core repos are Codex's **live** areas as of 2026-05-29:
- `arch-data`: 17 uncommitted changes on `codex/source-packages-pe-parity`.
- `microplex-us`: main +42 commits, multiple in-flight `codex/*` branches.

A rename now collides with that work and risks Codex's uncommitted changes. **A
freeze/sequence window (or landing Codex's current work first) is a prerequisite.**
Do not rename into a dirty Codex tree.

## Principles
- Per-repo PRs, each branched from `origin/main`, **each stage leaves everything
  working** (backward-compat aliases), and is reversible.
- Identifier-aware rename (word boundaries + known identifiers: `targets/arch.py`,
  `cosilico-arch`, `import arch`, bucket `"arch"`), with an explicit allowlist —
  never blanket `s/arch/ledger/`.
- **Data = copy-then-cutover, never rename/delete in place** (production microdata).

## Stages (dependency order; each non-breaking)
0. **Decide canonical names** (repo `ledger-data`? package `cosilico-ledger`?
   bucket `ledger`) and **open a coordination/freeze window with Codex**.
1. **arch-data (source of truth)**: introduce `ledger` module/package as canonical;
   keep `arch` as a thin re-export shim so existing importers don't break. Rename
   code/docs whole-word. Repo rename `arch-data → ledger-data` as a separate infra
   step (GitHub redirects old name).
2. **Dependents** (microplex, microplex-us, cosilico.ai): switch `import arch` →
   `import ledger`; rename internal refs (`targets/arch.py → targets/ledger.py`,
   console-script names, prose "Arch"→"Ledger"). Per-repo PRs landing *after* the
   stage-1 shim exists. microplex-us: do as a dedicated PR once Codex's current
   work merges (or have Codex own it, since it's their live area).
3. **R2 bucket**: `rclone/aws s3 cp arch → ledger` (copy, don't move) → enable
   dual-read in code → repoint writes to `ledger` → verify → deprecate `arch`.
4. **Artifact dirs**: the 49 `arch_*` dirs are regenerable build outputs; rename via
   copy + update any registry/manifest references, or regenerate. Low priority.
5. **Remove shims** + deprecate old repo/bucket once all dependents have cut over.

## What I will NOT do
- Blind/global `s/arch/ledger/` (corrupts substrings).
- Rename into Codex's dirty or fast-moving trees without a coordination window.
- Rename or delete the production R2 bucket in place (copy-then-cutover only).

## Status
- Plan recorded. Stage 0 (name decision + Codex coordination) is the gate.
- Nothing renamed yet — deliberately, pending the coordination window.
