from typing import Dict, List

from templates.base_template import build_common_user_prompt, build_system_prompt


APPLIANCE_CATEGORY = "appliance_repair"

APPLIANCE_SPECIFIC_PROMPT_INFO = (
	"Appliance-specific guidance to append to generic instructions:\n"
	"- Focus on common household appliances such as refrigerator, dishwasher, washer, dryer, microwave, and oven.\n"
	"- Insure to include a balanced mix of appliances so that one appliance type doesn't dominate the examples (e.g. don't just do 10 dryer issues).\n"
	"- Include model-agnostic troubleshooting steps before replacement (power check, filters, drain path, lint, door seals, leveling), and stop short of sealed-system, refrigerant, or other technician-only work.\n"
	"- Prefer homeowner-safe actions first; clearly escalate when high-voltage, refrigerant, gas, or sealed-system work is involved.\n"
	"- Include observable diagnostics in steps (error codes, sounds, vibration, heat level, standing water, cycle completion).\n"
	"- Tips must be NON-OBVIOUS appliance insights, NOT maintenance reminders. Examples of GOOD tips:\n"
	"  * For a leaking washer: 'Slow leaks often come from the door seal, not the hoses—check for visible wear or debris first'\n"
	"  * For a dryer not heating: 'If the drum turns but heat is off, the thermal fuse is likely blown—it's cheaper than calling a tech'\n"
	"  * For a dishwasher not cleaning: 'Hard water mineral buildup blocks spray arms more than dirty filters—run vinegar cycles monthly'\n"
	"  * For an oven not heating evenly: 'If one side heats more than the other, the heating element may be failing, not miscalibration'\n"
	"- AVOID: Generic checks, maintenance reminders, or redundant troubleshooting already in steps."
)


def build_appliance_repair_messages(count: int = 10) -> List[Dict[str, str]]:
	system_prompt = build_system_prompt(APPLIANCE_CATEGORY)
	user_prompt = build_common_user_prompt(
		category=APPLIANCE_CATEGORY,
		item_count=count,
		issue_type_example="dryer_not_heating",
	)
	user_prompt = f"{user_prompt}\n\n{APPLIANCE_SPECIFIC_PROMPT_INFO}"

	return [
		{"role": "system", "content": system_prompt},
		{"role": "user", "content": user_prompt},
	]
