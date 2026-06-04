# Dataset Prompt Changes - Iteration 8

This document records the prompt updates made to improve judge pass rates, especially for Q6 Tip Usefulness.

## Source Updated
- `templates/base_template.py`

## Summary
The dataset generation prompt was strengthened to reduce generic or obvious tips that were being flagged as unresolved by the judge.

## Changes Added To Prompt
A new section was added:

- `CRITICAL GUIDANCE ON TIPS (Q5 Tip Usefulness)`

This section now requires:
- Tips must be non-obvious and task-specific.
- Tips should add value beyond the numbered steps.
- Tips should explain practical insight beginners might miss.

This section now explicitly forbids:
- Generic maintenance reminders (for example: "check X periodically").
- Tips that repeat obvious troubleshooting already in steps.
- Irrelevant tool usage advice.
- Redundant information already covered in the answer section.

The section also includes concrete examples of strong tips for:
- Leaking faucet
- Clogged drain
- Electrical outlet intermittency
- Appliance odd-noise diagnosis

And adds a heuristic for quality:
- A good tip should answer: "Why does this work?" or "Why do beginners miss this?"

## Why This Was Added
Judge failures in prior runs frequently cited:
- "Tips are generic and obvious"
- "Tips are generic and not task-specific"

These prompt constraints were added to steer generation toward practical, non-obvious, high-signal tips that align with judge criteria.

## Notes
- This is an informational artifact for quick review.
- Canonical source-of-truth remains in Git history for `templates/base_template.py`.
