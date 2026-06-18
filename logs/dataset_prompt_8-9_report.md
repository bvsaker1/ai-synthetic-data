# Dataset Prompt Change Report: Iteration 8 -> 9

## Source
- Diff file: logs/dataset_prompt_8-9.diff

## Executive Summary
- Added repeated, explicit answer-completeness guidance across all categories.
- Strengthened scope boundaries for technician-only tasks.
- Added clearer escalation rules in electrical, plumbing, and HVAC contexts.

## Key Changes
| Area | What Changed | Intended Effect |
|---|---|---|
| Answer completeness | Added a CRITICAL block requiring tools to be referenced in answer/steps, closure of diagnostic loop, and brief rationale for fix. | Reduce vague or incomplete responses and improve closure quality. |
| Appliance scope | Added explicit boundary against sealed-system and refrigerant technician-only work. | Reduce unsafe/inappropriate DIY recommendations. |
| Electrical scope | Added damaged wires and explicit prohibition of framing wire replacement as DIY. | Improve safety and scope appropriateness. |
| Plumbing scope | Added explicit prohibition of DIY tank replacement. | Reduce hazardous or code-sensitive DIY guidance. |
| HVAC scope | Added explicit prohibition of implying DIY compressor/refrigerant replacement. | Improve tool realism and scope safety. |

## What A Third Party Should Notice
- Iteration 9 is materially stricter than iteration 8 on completeness and safety boundaries.
- The change pattern is broad and systematic (not one-off edits).

## Evidence
- See logs/dataset_prompt_8-9.diff for raw unified diff details.
