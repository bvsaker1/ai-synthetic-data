"""
Unit tests for judge_eval.py

Tests the LLM-as-judge evaluation pipeline including:
- Judge scoring from failed qualities
- Token rate limiting
- Retry logic and error handling
- JSON response parsing and normalization
- Dataset reading and item selection
"""

import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

import src.judge_eval as judge_eval
from src.judge_eval import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_MAX_COMPLETION_TOKENS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_TPM,
    TokenRateLimiter,
    _estimate_tokens,
    _normalize_judge_output,
    _parse_retry_after_seconds,
    _quality_key_to_label,
    build_iteration_path,
    build_judge_scoring,
    judge_one_item,
    load_judge_prompt_template,
    parse_args,
    read_dataset,
    select_items,
)


# Sample QA item for testing
SAMPLE_QA_ITEM = {
    "id": "qa_1",
    "category": "plumbing",
    "question": "How do I fix a leaky faucet?",
    "answer": "Turn off the water supply and replace the washers.",
    "equipment_problem": "leaky kitchen faucet",
    "tools_required": ["wrench", "screwdriver"],
    "steps": [
        "Turn off the water supply",
        "Disassemble the faucet handle",
        "Replace worn washers",
    ],
    "safety_info": "Ensure water is completely shut off before starting to avoid injury.",
    "tips": ["Keep track of small parts in a container", "Take photos before disassembly"],
}

# Valid judge response (RESOLVED)
VALID_JUDGE_RESPONSE_RESOLVED = {
    "resolved_label": 0,
    "status": "RESOLVED",
    "unresolved_reason": "",
    "failed_qualities": [],
    "judge_summary": "All criteria met successfully.",
}

# Valid judge response (UNRESOLVED)
VALID_JUDGE_RESPONSE_UNRESOLVED = {
    "resolved_label": 1,
    "status": "UNRESOLVED",
    "unresolved_reason": "Safety information is too generic.",
    "failed_qualities": [
        {
            "quality": "Q4 Safety Specificity",
            "why": "safety_info does not state a specific hazard or precaution for this specific task.",
        }
    ],
    "judge_summary": "Missing specific safety requirements for this repair.",
}


class TestQualityKeyToLabel:
    """Test quality key mapping"""

    def test_all_keys_present(self):
        """All Q1-Q8 should map to labels"""
        mapping = _quality_key_to_label()
        assert "Q1" in mapping
        assert "Q2" in mapping
        assert "Q3" in mapping
        assert "Q4" in mapping
        assert "Q5" in mapping
        assert "Q6" in mapping
        assert "Q7" in mapping
        assert "Q8" in mapping

    def test_q1_maps_to_coherence(self):
        """Q1 should map to answer_coherence"""
        mapping = _quality_key_to_label()
        assert mapping["Q1"] == "answer_coherence"

    def test_q4_maps_to_safety(self):
        """Q4 should map to safety_specificity"""
        mapping = _quality_key_to_label()
        assert mapping["Q4"] == "safety_specificity"


class TestBuildJudgeScoring:
    """Test judge scoring construction"""

    def test_all_pass_no_failures(self):
        """No failures should result in all scores = 1 and no flags set"""
        score = build_judge_scoring("qa_1", [])
        assert score["overall_failure"] == False
        assert score["quality_pass"] == True
        assert score["incomplete_answer"] == 0
        assert score["safety_violations"] == 0
        for quality_score in score["quality_scores"].values():
            assert quality_score == 1

    def test_single_failure_sets_score_to_0(self):
        """Single failure should set that quality score to 0"""
        failures = [{"quality": "Q1 Answer Coherence", "why": "Too brief"}]
        score = build_judge_scoring("qa_1", failures)
        assert score["quality_scores"]["answer_coherence"] == 0
        assert score["overall_failure"] == True
        assert score["quality_pass"] == False

    def test_multiple_failures(self):
        """Multiple failures should set corresponding scores"""
        failures = [
            {"quality": "Q1 Answer Coherence", "why": "Too brief"},
            {"quality": "Q4 Safety Specificity", "why": "Generic warning"},
        ]
        score = build_judge_scoring("qa_1", failures)
        assert score["quality_scores"]["answer_coherence"] == 0
        assert score["quality_scores"]["safety_specificity"] == 0
        assert score["overall_failure"] == True

    def test_failure_sets_flag(self):
        """Failures should set appropriate flags"""
        failures = [{"quality": "Q4 Safety Specificity", "why": "Generic"}]
        score = build_judge_scoring("qa_1", failures)
        assert score["safety_violations"] == 1

    def test_incomplete_answer_flag(self):
        """Q1/Q2 failures should set incomplete_answer flag"""
        failures = [{"quality": "Q1 Answer Coherence", "why": "Disjointed"}]
        score = build_judge_scoring("qa_1", failures)
        assert score["incomplete_answer"] == 1


class TestEstimateTokens:
    """Test token estimation"""

    def test_empty_string(self):
        """Empty string should return at least 1 token"""
        result = _estimate_tokens("")
        assert result == 1

    def test_short_string(self):
        """Short string should return at least 1 token"""
        result = _estimate_tokens("hi")
        assert result >= 1

    def test_longer_string(self):
        """Longer strings should scale with length"""
        result1 = _estimate_tokens("test")
        result2 = _estimate_tokens("test" * 10)
        assert result2 > result1


class TestParseRetryAfterSeconds:
    """Test retry-after parsing"""

    def test_seconds_format(self):
        """Parse seconds format correctly"""
        error_text = "Rate limit: Please try again in 6.165s"
        result = _parse_retry_after_seconds(error_text)
        assert abs(result - 6.165) < 0.01

    def test_milliseconds_format(self):
        """Parse milliseconds format correctly"""
        error_text = "Rate limit: Please try again in 262.5ms"
        result = _parse_retry_after_seconds(error_text)
        assert abs(result - 0.2625) < 0.001

    def test_no_match_returns_zero(self):
        """No match should return 0.0"""
        error_text = "Some other error"
        result = _parse_retry_after_seconds(error_text)
        assert result == 0.0


class TestTokenRateLimiter:
    """Test token rate limiting"""

    def test_initialization(self):
        """Initialize with parameters"""
        limiter = TokenRateLimiter(max_tokens_per_window=1000)
        assert limiter.max_tokens_per_window == 1000

    def test_negative_tokens_raises(self):
        """Negative tokens should raise ValueError"""
        with pytest.raises(ValueError):
            TokenRateLimiter(max_tokens_per_window=-1)

    def test_reserve_within_limit(self):
        """Reserve within limit succeeds"""
        limiter = TokenRateLimiter(max_tokens_per_window=1000)
        limiter.reserve(500)
        assert limiter._tokens_in_window == 500

    def test_multiple_reserves(self):
        """Multiple reserves accumulate"""
        limiter = TokenRateLimiter(max_tokens_per_window=1000)
        limiter.reserve(300)
        limiter.reserve(300)
        limiter.reserve(400)
        assert limiter._tokens_in_window == 1000


class TestNormalizeJudgeOutput:
    """Test judge response normalization"""

    def test_valid_resolved_response(self):
        """Valid resolved response passes through"""
        result = _normalize_judge_output(VALID_JUDGE_RESPONSE_RESOLVED)
        assert result["resolved_label"] == 0
        assert result["status"] == "RESOLVED"

    def test_valid_unresolved_response(self):
        """Valid unresolved response passes through"""
        result = _normalize_judge_output(VALID_JUDGE_RESPONSE_UNRESOLVED)
        assert result["resolved_label"] == 1
        assert result["status"] == "UNRESOLVED"
        assert len(result["failed_qualities"]) > 0

    def test_status_string_resolved_sets_label(self):
        """status='RESOLVED' sets label"""
        response = {
            "status": "RESOLVED",
            "failed_qualities": [],
            "judge_summary": "OK",
        }
        result = _normalize_judge_output(response)
        assert result["resolved_label"] == 0

    def test_unresolved_without_reason_generates_one(self):
        """UNRESOLVED without reason generates one"""
        response = {
            "resolved_label": 1,
            "failed_qualities": [{"quality": "Q1", "why": "bad"}],
            "judge_summary": "Bad",
        }
        result = _normalize_judge_output(response)
        assert result["unresolved_reason"] != ""


class TestReadDataset:
    """Test dataset reading"""

    def test_read_jsonl_file(self):
        """Read JSONL file format"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.jsonl"
            path.write_text(
                json.dumps(SAMPLE_QA_ITEM) + "\n" +
                json.dumps(SAMPLE_QA_ITEM) + "\n"
            )
            items = read_dataset(str(path))
            assert len(items) == 2


class TestPromptTemplate:
    """Test external judge prompt template loading"""

    def test_load_judge_prompt_template(self):
        """Valid prompt template should load and validate"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "judge_prompt_7.json"
            path.write_text(
                json.dumps(
                    {
                        "system_prompt": "system text",
                        "user_prompt_template": "Prompt body\n__ITEM_JSON__",
                    }
                ),
                encoding="utf-8",
            )
            prompt_template = load_judge_prompt_template(str(path))
            assert prompt_template["system_prompt"] == "system text"
            assert "__ITEM_JSON__" in prompt_template["user_prompt_template"]

    def test_load_judge_prompt_template_requires_placeholder(self):
        """Prompt template must include the item placeholder"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "judge_prompt_7.json"
            path.write_text(
                json.dumps(
                    {
                        "system_prompt": "system text",
                        "user_prompt_template": "Prompt body without placeholder",
                    }
                ),
                encoding="utf-8",
            )
            with pytest.raises(ValueError):
                load_judge_prompt_template(str(path))


class TestParseArgs:
    """Test command line parsing for iteration-aware defaults"""

    def test_default_iteration_builds_default_paths(self, monkeypatch):
        """Default iteration should produce correct path extensions"""
        monkeypatch.setattr("sys.argv", ["judge_eval.py"])
        args = parse_args()
        assert args.iteration == judge_eval.get_iteration()
        assert args.dataset == build_iteration_path("diy_dataset", args.iteration, subdir="data")
        assert args.output == build_iteration_path("judge_labels", args.iteration, subdir="labels")
        assert args.prompt_template == build_iteration_path("judge_prompt", args.iteration, extension=".json", subdir="templates")

    def test_custom_iteration_updates_default_paths(self, monkeypatch):
        """Custom iteration should flow into default dataset/output/prompt paths"""
        monkeypatch.setattr("sys.argv", ["judge_eval.py", "--iteration", "7"])
        args = parse_args()
        assert args.iteration == "7"
        assert args.dataset.endswith("diy_dataset_7.jsonl")
        assert args.output.endswith("labels/judge_labels_7.jsonl")
        assert args.prompt_template.endswith("judge_prompt_7.json")

    def test_read_json_file(self):
        """Read JSON array file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.json"
            path.write_text(json.dumps([SAMPLE_QA_ITEM, SAMPLE_QA_ITEM]))
            items = read_dataset(str(path))
            assert len(items) == 2

    def test_skip_empty_lines(self):
        """Skip empty lines in JSONL"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.jsonl"
            path.write_text(
                json.dumps(SAMPLE_QA_ITEM) + "\n" +
                "\n" +
                json.dumps(SAMPLE_QA_ITEM) + "\n"
            )
            items = read_dataset(str(path))
            assert len(items) == 2


class TestSelectItems:
    """Test item selection logic"""

    def test_sequential_all_items(self):
        """Sequential mode returns all items"""
        items = [{"id": "1"}, {"id": "2"}, {"id": "3"}]
        result = select_items(items, items_to_test=None, random_mode=False)
        assert len(result) == 3

    def test_sequential_limited(self):
        """Sequential mode with limit returns first N"""
        items = [{"id": "1"}, {"id": "2"}, {"id": "3"}]
        result = select_items(items, items_to_test=2, random_mode=False)
        assert len(result) == 2

    def test_random_mode_no_limit_returns_one(self):
        """Random mode without limit returns 1"""
        items = [
            {"id": "1", "category": "appliance"},
            {"id": "2", "category": "electrical"},
            {"id": "3", "category": "plumbing"},
            {"id": "4", "category": "general_home"},
            {"id": "5", "category": "hvac"},
        ]
        result = select_items(items, items_to_test=None, random_mode=True)
        assert len(result) == 1

    def test_random_mode_balances_categories_with_limit(self):
        """Random mode should select an evenly distributed category mix"""
        items: List[Dict[str, Any]] = []
        for category in judge_eval.CATEGORIES:
            for index in range(4):
                items.append({"id": f"{category}_{index}", "category": category})

        result = select_items(items, items_to_test=10, random_mode=True)

        counts = {category: 0 for category in judge_eval.CATEGORIES}
        for row in result:
            counts[str(row.get("category", ""))] += 1

        assert len(result) == 10
        assert all(count == 2 for count in counts.values())

    def test_random_mode_missing_required_category_raises(self):
        """Balanced random mode should fail when a required category is missing"""
        items = [
            {"id": "1", "category": "appliance"},
            {"id": "2", "category": "electrical"},
            {"id": "3", "category": "plumbing"},
            {"id": "4", "category": "general_home"},
        ]

        with pytest.raises(ValueError, match="required categories are missing"):
            select_items(items, items_to_test=4, random_mode=True)

    def test_empty_items_returns_empty(self):
        """Empty items list returns empty"""
        result = select_items([], items_to_test=None, random_mode=False)
        assert result == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
