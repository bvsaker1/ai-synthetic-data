from typing import Dict, List

from base_template import build_common_user_prompt, build_system_prompt


PLUMBING_CATEGORY = "plumbing_repair"

PLUMBING_SPECIFIC_PROMPT_INFO = (
	"Plumbing-specific guidance to append to generic instructions:\n"
	"- Example topics: leaks, clogs, shutoff valves, traps, toilet fill/flush issues, fixture replacement, water heaters.\n"
	"- Include when to escalate to licensed plumber (main line, sewer, hidden leaks behind walls, code-required work)."
)


def build_plumbing_repair_messages(count: int = 10) -> List[Dict[str, str]]:
	system_prompt = build_system_prompt(PLUMBING_CATEGORY)
	user_prompt = build_common_user_prompt(
		category=PLUMBING_CATEGORY,
		item_count=count,
		issue_type_example="faucet_leak",
	)
	user_prompt = f"{user_prompt}\n\n{PLUMBING_SPECIFIC_PROMPT_INFO}"

	return [
		{"role": "system", "content": system_prompt},
		{"role": "user", "content": user_prompt},
	]
