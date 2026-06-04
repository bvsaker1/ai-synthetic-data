def build_system_prompt(category: str) -> str:
	return (
		f"You generate high-quality DIY dataset items for {category} in strict JSON. "
		"Output must be valid JSON only with keys: question, answer, equipment_problem, "
		"tools_required, steps, safety_info, tips. "
	)


def build_common_user_prompt(
	category: str,
	item_count: int,
	issue_type_example: str,
) -> str:
	return (
		f"Create {item_count} unique {category} QA items.\n\n"
		"Validation rules to satisfy in the output schema:\n"
		"1) question and answer are non-empty strings\n"
		"2) steps contains at least 3 items\n"
		"3) tools_required contains at least 1 item\n"
		"4) tips contains at least 1 item\n"
		"5) safety_info is present and non-empty\n\n"
		"LLM-evaluated failure modes that should be proactively avoided in generation:\n"
		"Q1 Answer Coherence: answer must be a natural, unified narrative, not a stitched list.\n"
		"Q2 Step Actionability: steps must be concrete and specific enough for beginners; avoid vague phrasing.\n"
		"Q3 Tool Realism: tools should be typical homeowner tools available in general hardware stores, typically under $50 each.\n"
		"Q4 Safety Specificity: safety_info must name the specific hazard and specific precaution; avoid generic warnings.\n"
		"Q5 Tip Usefulness: each tip must be non-obvious and add value beyond the steps.\n"
		"Q6 Problem-Answer Alignment: answer must directly solve the exact problem in equipment_problem.\n"
		"Q7 Appropriate Scope: keep within realistic DIY scope; for risky repairs, clearly recommend professional help.\n"
		f"Q8 Category Accuracy: content must match {category}.\n\n"
		"Key principles for dataset creation:\n"
		"- Provide a substantial narrative answer, typically around 700-1300 characters.\n"
		"- Safety guidance must be task-specific and hazard-specific.\n"
		"- Tips should include practical, non-obvious insights a beginner might not know.\n"
		"- Steps should include observable outcomes, quantities, or thresholds when relevant.\n\n"
		"CRITICAL GUIDANCE ON TIPS (Q5 Tip Usefulness):\n"
		"Tips MUST be non-obvious, task-specific insights that beginners don't naturally know.\n"
		"AVOID these common tip mistakes:\n"
		"- DO NOT suggest generic maintenance checks (e.g., 'check X periodically', 'inspect Y monthly')\n"
		"- DO NOT suggest obvious troubleshooting steps that are already in the main steps\n"
		"- DO NOT suggest irrelevant tools (e.g., using a level for painting or weather stripping)\n"
		"- DO NOT suggest redundant information from the answer section\n"
		"GOOD tip examples (what to aim for):\n"
		"- For a leaking faucet: 'Before replacing the entire cartridge, try tightening the packing nut under the handle—many leaks stop there'\n"
		"- For a clogged drain: 'Use a plunger with standing water for suction; dry plunging is less effective'\n"
		"- For an electrical outlet: 'If power is intermittent, the breaker may be failing, not the outlet'\n"
		"- For appliance repair: 'Listen for odd sounds during operation—grinding means the motor bearing is failing, not just a loose part'\n"
		"GOOD tips share:  specific conditions, counterintuitive insights, warnings about common false-fixes, or non-obvious root causes.\n"
		"Each tip should answer a 'Why does this work?' or 'Why do beginners miss this?' question.\n\n"
		"Output JSON schema:\n"
		"{\n"
		"  \"question\": \"...\",\n"
		"  \"answer\": \"...\",\n"
		"  \"equipment_problem\": \"...\",\n"
		"  \"tools_required\": [\"...\"],\n"
		"  \"steps\": [\"...\", \"...\", \"...\"],\n"
		"  \"safety_info\": \"...\",\n"
		"  \"tips\": [\"...\"]\n"
		"}\n\n"
		"Return JSON only. No markdown."
	)
