# Reviews

This directory is the durable handoff surface for Claude/Codex review work.

## Steady-state workflow

1. Codex writes the current review request to:
   - `PENDING_CLAUDE_REVIEW.md`
2. The user gives Claude a short instruction such as:
   - `Please execute the pending review request in /Users/maxghenis/PolicyEngine/microplex-us/reviews/PENDING_CLAUDE_REVIEW.md`
3. Claude writes the full review to a dated file in this directory.
4. Claude appends a short summary to:
   - `/Users/maxghenis/PolicyEngine/microplex-us/_BUILD_LOG.md`

## File roles

- `PENDING_CLAUDE_REVIEW.md`
  - current review request only
- `YYYY-MM-DD-*.md`
  - full saved review outputs

Keep `_BUILD_LOG.md` short. Full reviews belong here, not in the log.
