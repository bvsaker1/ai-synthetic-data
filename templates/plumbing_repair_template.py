from typing import Dict, List

from templates.base_template import build_common_user_prompt, build_system_prompt


PLUMBING_CATEGORY = "plumbing_repair"

PLUMBING_SPECIFIC_PROMPT_INFO = (
	"Plumbing-specific guidance to append to generic instructions:\n"
	"- Example topics: leaks, clogs, shutoff valves, traps, toilet fill/flush issues, fixture replacement, water heaters.\n"
	"- Include when to escalate to licensed plumber (main line, sewer, hidden leaks behind walls, code-required work), and do not suggest DIY tank replacement.\n"
	"- Tips must be NON-OBVIOUS plumbing insights, NOT maintenance checklists. Examples of GOOD tips:\n"
	"  * For a slow drain: 'If plunging the sink doesn't work but the toilet flushes normally, the clog is in the vent stack, not the drain branch'\n"
	"  * For a leaking faucet: 'If tightening the packing nut doesn't stop dripping, the cartridge seal is worn—replacing the cartridge is cheaper than a new faucet'\n"
	"  * For a running toilet: 'Listen closely—if water hisses, the fill valve is leaking; if it gurgles, the flapper is stuck'\n"
	"  * For low water pressure: 'Check the shutoff valve first—it's often partially closed after work; sediment in aerators is second'\n"
	"- AVOID: Generic maintenance checks, obvious preventive actions, or redundant drain/leak troubleshooting."
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
