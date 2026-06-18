# Dataset Prompt Change Report: Iteration 10 -> 11

## Source
- Diff file: logs/dataset_prompt_10-11.diff

## Executive Summary
- Major rubric normalization and terminology rewrite across all categories.
- Reframed dimensions toward project label schema (completeness, safety, scope, context, tips).
- Added stronger explicit requirements that tools be mentioned in answer text.

## Key Changes
| Area | What Changed | Intended Effect |
|---|---|---|
| Rubric naming | Replaced older dimension labels (coherence/actionability/problem alignment) with completeness/safety/scope/context mapping. | Align generation guidance with downstream judging/label schema. |
| Safety language | Changed from generic "hazard-specific" phrasing to explicit "name hazard + precaution" framing. | Improve safety specificity and consistency. |
| Tool usage requirement | Strengthened to require tool references in answer, not only in tools list. | Reduce disconnect between tool list and narrative steps. |
| Tip quality constraints | Tightened relevance and anti-obviousness with more examples. | Improve tip usefulness quality and reduce noisy tips. |
| Scope constraints | Additional category-specific scope prohibitions and clarifications. | Reduce out-of-scope DIY guidance. |

## Risk Note
- This iteration is a broad rewrite. Improvements may be accompanied by metric volatility due to simultaneous multi-dimension changes.

## What A Third Party Should Notice
- Iteration 11 is not incremental; it is a structural prompt policy shift.

## Evidence
- See logs/dataset_prompt_10-11.diff for raw unified diff details.
