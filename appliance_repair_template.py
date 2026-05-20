from typing import Dict, List

from base_template import build_common_user_prompt, build_system_prompt


APPLIANCE_CATEGORY = "appliance_repair"

APPLIANCE_SPECIFIC_PROMPT_INFO = (
	"Appliance-specific guidance to append to generic instructions:\n"
	"- Focus on common household appliances such as refrigerator, dishwasher, washer, dryer, microwave, and oven.\n"
	"- Insure to include a balanced mix of appliances so that one appliance type doesn't dominate the examples (e.g. don't just do 10 dryer issues).\n"
	"- Include model-agnostic troubleshooting steps before replacement (power check, filters, drain path, lint, door seals, leveling).\n"
	"- Prefer homeowner-safe actions first; clearly escalate when high-voltage, refrigerant, gas, or sealed-system work is involved.\n"
	"- Include observable diagnostics in steps (error codes, sounds, vibration, heat level, standing water, cycle completion).\n"
	"- Tips should include appliance-specific preventive maintenance (clean condenser coils, inspect hoses, clean filters, avoid overload)."
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
