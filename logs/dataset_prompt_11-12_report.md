# Dataset Prompt Change Report: Iteration 11 -> 12

## Source
- Diff file: logs/dataset_prompt_11-12.diff

## Executive Summary
- Focused refinement on tip usefulness section labeling and formatting consistency.
- Added explicit anti-pattern: do not repeat safety info as tips.
- Tightened electrical scope wording (panel/circuit-breaker replacement out of scope).

## Key Changes
| Area | What Changed | Intended Effect |
|---|---|---|
| Tip usefulness section | Relabeled from Q5 to Q6 and reformatted bullets consistently. | Reduce rubric ambiguity and improve prompt readability. |
| Tip anti-patterns | Added explicit prohibition against repeating safety guidance in tips. | Improve non-redundant, actionable tip quality. |
| Electrical scope | Revised statement to place panel and breaker replacement out of scope. | Reduce unsafe electrical DIY guidance. |

## What A Third Party Should Notice
- Iteration 12 is mostly a quality-control pass over wording consistency and stricter scope/safety clarity, not a full redesign.

## Evidence
- See logs/dataset_prompt_11-12.diff for raw unified diff details.
