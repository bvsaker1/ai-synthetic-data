# Dataset Prompt Change Report: Iteration 12 -> 13

## Source
- Diff file: logs/dataset_prompt_12-13.diff

## Executive Summary
- Small targeted edits focused on explicit tool mention and scope/safety specificity.

## Key Changes
| Area | What Changed | Intended Effect |
|---|---|---|
| Tool mention | Added explicit reminder: wrench must be mentioned in answer when listed in tools. | Improve answer completeness and tool-answer alignment. |
| Safety specificity (Q2) | Added explicit rule that safety hazard must be specific and not vague. | Improve safety specificity quality signal. |
| Scope appropriateness (Q4) | Added explicit rule: refrigerant checking is out of scope and professional-only. | Reduce unsafe HVAC DIY recommendations. |

## What A Third Party Should Notice
- Iteration 13 is a small, high-precision refinement iteration.

## Evidence
- See logs/dataset_prompt_12-13.diff for raw unified diff details.
