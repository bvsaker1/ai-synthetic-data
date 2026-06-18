from typing import Dict, List

from templates.base_template import build_common_user_prompt, build_system_prompt


HVAC_CATEGORY = "hvac_maintenance"

HVAC_SPECIFIC_PROMPT_INFO = (
	"HVAC-specific guidance to append to generic instructions:\n"
	"- Example topics: filter changes, thermostat checks, condensate drain cleaning, airflow balancing, furnace not running, ac not running.\n"
	"- Include when to escalate to certified technician (refrigerant handling, compressor faults, electrical control board issues), and never imply compressor or refrigerant replacement is DIY.\n"
	"- Tips must be NON-OBVIOUS HVAC insights, NOT maintenance reminders. Examples of GOOD tips:\n"
	"  * For uneven heating/cooling: 'Closed supply vents in unused rooms can starve other areas—try balancing vents instead of adjusting thermostat'\n"
	"  * For a furnace not running: 'The pilot light or ignition sensor may just be dirty—cleaning it is faster and cheaper than service calls'\n"
	"  * For low AC cooling: 'A frozen evaporator coil (not low refrigerant) causes weak airflow—check the condensate drain for blockages first'\n"
	"  * For a noisy blower: 'Rattling usually means a loose access panel or duct, not a failing motor—tighten fasteners before calling for service'\n"
	"- AVOID: Generic filter reminders, obvious maintenance checklists, or redundant diagnostic steps."
	"- CRITICAL GUIDANCE ON TOOL REALISM (Q3 Tool Realism):\n"
	"  * A thermostat or pilot light are not normal homeowner tools and cost more than $50.\n"
	"  * A refrigerant leak detector is not a normal homeowner tool and it's use is out of scope.\n"
	"- CRITICAL GUIDANCE ON TIP USEFULNESS (Q6 Tip Usefulness):\n"
	"  * Tips MUST NOT be vague and obvious.\nExamples to follow:\n"
	"  * For a question on how to fix a faulty thermostat DO NOT give tips saying that a faulty thermostat can cause uneven heating.\n"
	"  * For a question on why furnace is not turning on, a tip saying that a faulty thermostat can cause the furnace not to turn on is obvious and should be in the answer not the tip.\n"
	"  * For a question on why furnace is not heating, cleaning the ignition sensor or checking for debris on the pilot should not be tips if is something already in the answer with instructions to check.\n"
	"- SAFETY SPECIFICATION (Q2): Safety hazard must be specific and NOT vague.\n"
	"- SCOPE APPROPRIATENESS (Q4): Refrigerant checking out of scope and should be done by a professional."
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
