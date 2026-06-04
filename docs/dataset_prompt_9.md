# Dataset Prompt Changes - Iteration 9

This document records the prompt updates made to improve judge pass rates for the next run, with emphasis on Q1 Answer Coherence / Answer Completeness.

## Source Updated
- [templates/base_template.py](../templates/base_template.py)
- [templates/appliance_repair_template.py](../templates/appliance_repair_template.py)
- [templates/electrical_repair_template.py](../templates/electrical_repair_template.py)
- [templates/plumbing_repair_template.py](../templates/plumbing_repair_template.py)
- [templates/gen_home_repair_template.py](../templates/gen_home_repair_template.py)
- [templates/hvac_maintenance_template.py](../templates/hvac_maintenance_template.py)

## Summary
The prompt was strengthened so generated answers read like complete repair guidance instead of partial troubleshooting notes.

## Changes Added To Prompt

A new shared section was added to the base template:

- If tools_required lists specific tools, explicitly reference them in the answer or steps.
- Do not mention tools only in tools_required; weave the critical tools into the repair narrative.
- Answers must name the likely cause, the fix, how to verify success, and when to stop.
- Answers should not leave the user with only symptoms or a tool list.
- Answers should briefly explain why the fix works when relevant.
- Answers should stay concise but still feel like a finished repair plan.

Category-specific reinforcement was added for:

- Appliance repair: require the likely failing subsystem or component, the repair, the confirmation check, and explicit use of the listed tools in the steps.
- Electrical repair: require the circuit/device/connection point, the fix, the confirmation test, and explicit mention of safe homeowner tools.
- Plumbing repair: require the likely part of the water path, the proof the repair worked, and explicit use of the listed tools in the steps.
- HVAC repair: require the likely subsystem, the verification step after the fix, and explicit mention of safe homeowner tools.
- General home repair: require the material or hardware, the corrective action, and how to verify the result.

## Why This Was Added
Recent analysis showed elevated failures for answer completeness/coherence, especially:

- Appliance answer completeness at 50%
- Electrical, plumbing, and HVAC answer completeness at 25%

The log showed that the repeated Q1 misses were usually about omitted tool references inside the answer/steps, not just short answers. It also showed Q4/Q7 drift toward unsafe DIY scope in electrical, plumbing, and HVAC when wire, tank, refrigerant, or compressor work was implied.

## Intended Effect
These changes should push generation toward answers that:

- State the most likely cause early
- Explain the actual repair path instead of only the symptom
- Include a clear verification step
- End with a sensible escalation point when DIY should stop

## Environment Update
- Set [../.env](../.env) to `ITERATION=9` for the next generation and evaluation run.

## Notes
- This is an informational artifact for quick review.
- Canonical source-of-truth remains in the template files above.