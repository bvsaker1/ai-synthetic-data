from typing import Dict, List

from base_template import build_common_user_prompt, build_system_prompt


GENERAL_HOME_REPAIR_CATEGORY = "general_home_repair"

GENERAL_HOME_REPAIR_SPECIFIC_PROMPT_INFO = (
    "General home-repair-specific guidance to append to generic instructions:\n"
	"- Example topics: minor drywall damage, trim fixes, caulking, squeaky hinges, weatherstripping, painting touch-ups, door adjustments.\n"
	"- Include escalation when repairs involve structural, electrical-panel, gas, or major plumbing risks."
)


def build_general_home_repair_messages(count: int = 10) -> List[Dict[str, str]]:
	system_prompt = build_system_prompt(GENERAL_HOME_REPAIR_CATEGORY)
	user_prompt = build_common_user_prompt(
		category=GENERAL_HOME_REPAIR_CATEGORY,
		item_count=count,
		issue_type_example="wall_crack_patch",
	)
	user_prompt = f"{user_prompt}\n\n{GENERAL_HOME_REPAIR_SPECIFIC_PROMPT_INFO}"

	return [
		{"role": "system", "content": system_prompt},
		{"role": "user", "content": user_prompt},
	]
