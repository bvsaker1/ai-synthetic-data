from typing import Dict, List

from base_template import build_common_user_prompt, build_system_prompt


ELECTRICAL_CATEGORY = "electrical_repair"

ELECTRICAL_SPECIFIC_PROMPT_INFO = (
    "Electrical-specific guidance to append to generic instructions:\n"
	"- You are an expert home electrician providing detailed, step-by-step repair instructions for common household electrical issues.\n"
	"- Your guidance should be safe, practical, and accessible for a homeowner with basic DIY skills.\n"
	"Always prioritize safety and include clear escalation points to licensed electricians when repairs involve high-voltage, complex wiring, or potential code violations.\n"
	"- Example topics: non-working outlets, tripped breakers, switch replacement, GFCI resets.\n"
	"- Include strict safety escalation for panel work, unknown wiring, aluminum wiring, or signs of overheating/arcing."
)


def build_electrical_repair_messages(count: int = 10) -> List[Dict[str, str]]:
	system_prompt = build_system_prompt(ELECTRICAL_CATEGORY)
	user_prompt = build_common_user_prompt(
		category=ELECTRICAL_CATEGORY,
		item_count=count,
		issue_type_example="outlet_not_working",
	)
	user_prompt = f"{user_prompt}\n\n{ELECTRICAL_SPECIFIC_PROMPT_INFO}"

	return [
		{"role": "system", "content": system_prompt},
		{"role": "user", "content": user_prompt},
	]
