from typing import Dict, List

from base_template import build_common_user_prompt, build_system_prompt


HVAC_CATEGORY = "hvac_maintenance"

HVAC_SPECIFIC_PROMPT_INFO = (
	"HVAC-specific guidance to append to generic instructions:\n"
	"- Example topics: filter changes, thermostat checks, condensate drain cleaning, airflow balancing, furnace not running, ac not running.\n"
	"- Include when to escalate to certified technician (refrigerant handling, compressor faults, electrical control board issues)."
)


def build_hvac_maintenance_messages(count: int = 10) -> List[Dict[str, str]]:
	system_prompt = build_system_prompt(HVAC_CATEGORY)
	user_prompt = build_common_user_prompt(
		category=HVAC_CATEGORY,
		item_count=count,
		issue_type_example="air_filter_replacement",
	)
	user_prompt = f"{user_prompt}\n\n{HVAC_SPECIFIC_PROMPT_INFO}"

	return [
		{"role": "system", "content": system_prompt},
		{"role": "user", "content": user_prompt},
	]
