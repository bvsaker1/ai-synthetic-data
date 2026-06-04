from typing import Dict, List

from templates.base_template import build_common_user_prompt, build_system_prompt


ELECTRICAL_CATEGORY = "electrical_repair"

ELECTRICAL_SPECIFIC_PROMPT_INFO = (
    "Electrical-specific guidance to append to generic instructions:\n"
	"- You are an expert home electrician providing detailed, step-by-step repair instructions for common household electrical issues.\n"
	"- Your guidance should be safe, practical, and accessible for a homeowner with basic DIY skills.\n"
	"Always prioritize safety and include clear escalation points to licensed electricians when repairs involve high-voltage, complex wiring, or potential code violations.\n"
	"- Example topics: non-working outlets, tripped breakers, switch replacement, GFCI resets.\n"
	"- Include strict safety escalation for panel work, unknown wiring, aluminum wiring, damaged wires, or signs of overheating/arcing; do not frame wire replacement as a DIY step.\n"
	"- Tips must be NON-OBVIOUS electrical insights, NOT safety reminders. Examples of GOOD tips:\n"
	"  * For a dead outlet: 'Before replacing the outlet, check if a nearby GFCI outlet is tripped—it may control this outlet from elsewhere'\n"
	"  * For flickering lights: 'Intermittent flickering often means a loose connection at the breaker, not the light fixture itself'\n"
	"  * For tripped breaker: 'If it trips immediately, there's a short circuit; if it trips after running, the circuit is overloaded'\n"
	"  * For a dimmer switch: 'Not all LED bulbs work with dimmers—look for \"dimmable\" rating to avoid strobing or failure'\n"
	"- AVOID: Generic safety warnings, basic tool reminders, or redundant troubleshooting steps."
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
