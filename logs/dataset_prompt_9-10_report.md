# Dataset Prompt Change Report: Iteration 9 -> 10

## Source
- Diff file: logs/dataset_prompt_9-10.diff

## Executive Summary
- Added targeted corrective guidance for previously weak dimensions.
- Tightened electrical scope constraints and HVAC tool/tip constraints.
- Added plumbing completeness reminder to mention plumber's tape and tool usage.

## Key Changes
| Area | What Changed | Intended Effect |
|---|---|---|
| Electrical scope (Q4) | Added explicit limits for panel replacement, new circuits, and whole-room wiring. | Reduce out-of-scope electrical recommendations. |
| Plumbing completeness (Q1) | Added explicit plumber's tape mention requirement and stronger tool mention requirement. | Improve answer completeness and practical execution quality. |
| HVAC tool realism (Q3) | Added explicit disallow list examples (thermostat/pilot light/refrigerant leak detector as tools). | Reduce unrealistic tool recommendations. |
| HVAC tip usefulness (Q6) | Added anti-vagueness examples to prevent obvious or duplicated tips. | Improve tip usefulness signal quality. |

## What A Third Party Should Notice
- Iteration 10 focuses on corrective, category-specific quality failures rather than broad policy changes.
- Most edits are tightly aligned to known regression areas.

## Evidence
- See logs/dataset_prompt_9-10.diff for raw unified diff details.
