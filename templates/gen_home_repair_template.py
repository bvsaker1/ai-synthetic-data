from typing import Dict, List

from templates.base_template import build_common_user_prompt, build_system_prompt


GENERAL_HOME_REPAIR_CATEGORY = "general_home_repair"

GENERAL_HOME_REPAIR_SPECIFIC_PROMPT_INFO = (
    "General home-repair-specific guidance to append to generic instructions:\n"
	"- Example topics: minor drywall damage, trim fixes, caulking, squeaky hinges, weatherstripping, painting touch-ups, door adjustments.\n"
	"- Include escalation when repairs involve structural, electrical-panel, gas, or major plumbing risks.\n"
	"- Tips must be NON-OBVIOUS general home repair insights, NOT irrelevant tool advice. Examples of GOOD tips:\n"
	"  * For drywall patching: 'Small holes should use spackle, but large holes need mesh tape first—skipping tape causes future cracking'\n"
	"  * For painting: 'Primer isn't just for new walls—use it on stains and water marks to prevent bleed-through, not for evenness'\n"
	"  * For weatherstripping: 'Felt wears faster than rubber—use rubber or EPDM for doors; felt is fine for windows only'\n"
	"  * For squeaky hinges: 'If oil doesn't stop squeaking, the pin is bent—remove and straighten it on a hard surface'\n"
	"  * For caulking: 'Silicone caulk is waterproof but too rigid outdoors—use paintable acrylic for trim; silicone for bathrooms only'\n"
	"- AVOID: Irrelevant tool recommendations (e.g., level for painting/weatherstripping), obvious maintenance checks, or redundant repair steps."
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
