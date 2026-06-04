"""
Unit tests for dataset_generation.py

Tests the generation pipeline including:
- LLM batch calls with Instructor validation
- Item collection from templates
- Raw dataset storage
- Pydantic schema validation
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

import src.dataset_generation as dataset_generation
from src.dataset_generation import (
    MAX_LLM_CALLS_PER_ATTEMPT,
    RepairQABatchModel,
    RepairQAModel,
    call_llm_typed_batch,
    generate_candidate_items,
    generate_items_from_template,
    parse_args,
    save_raw_dataset_append,
)
from src.logging_utils import JsonEventLogger


# Sample valid QA item
SAMPLE_QA_ITEM = {
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


class TestRepairQAModel:
    """Test Pydantic validation for RepairQAModel"""

    def test_valid_item(self):
        """Valid item should pass validation"""
        item = RepairQAModel(**SAMPLE_QA_ITEM)
        assert item.question == SAMPLE_QA_ITEM["question"]
        assert item.answer == SAMPLE_QA_ITEM["answer"]
        assert item.judge_scoring == ""

    def test_missing_required_field(self):
        """Missing required field should raise ValidationError"""
        incomplete_item = SAMPLE_QA_ITEM.copy()
        del incomplete_item["question"]
        with pytest.raises(ValidationError):
            RepairQAModel(**incomplete_item)

    def test_empty_string_field(self):
        """Empty string in required field should fail"""
        invalid_item = SAMPLE_QA_ITEM.copy()
        invalid_item["question"] = ""
        with pytest.raises(ValidationError):
            RepairQAModel(**invalid_item)

    def test_steps_minimum_length(self):
        """Steps must have at least 3 items"""
        invalid_item = SAMPLE_QA_ITEM.copy()
        invalid_item["steps"] = ["Step 1", "Step 2"]
        with pytest.raises(ValidationError):
            RepairQAModel(**invalid_item)

    def test_tools_required_minimum_length(self):
        """tools_required must have at least 1 item"""
        invalid_item = SAMPLE_QA_ITEM.copy()
        invalid_item["tools_required"] = []
        with pytest.raises(ValidationError):
            RepairQAModel(**invalid_item)

    def test_extra_fields_ignored(self):
        """Extra fields should be ignored due to model_config"""
        item_with_extra = SAMPLE_QA_ITEM.copy()
        item_with_extra["extra_field"] = "should be ignored"
        item = RepairQAModel(**item_with_extra)
        assert not hasattr(item, "extra_field")


class TestRepairQABatchModel:
    """Test batch model for collecting multiple items"""

    def test_valid_batch(self):
        """Valid batch with multiple items"""
        batch_data = {"items": [SAMPLE_QA_ITEM, SAMPLE_QA_ITEM]}
        batch = RepairQABatchModel(**batch_data)
        assert len(batch.items) == 2

    def test_batch_minimum_length(self):
        """Batch must have at least 1 item"""
        batch_data = {"items": []}
        with pytest.raises(ValidationError):
            RepairQABatchModel(**batch_data)

    def test_invalid_item_in_batch(self):
        """Invalid item in batch should raise ValidationError"""
        incomplete_item = SAMPLE_QA_ITEM.copy()
        del incomplete_item["question"]
        batch_data = {"items": [SAMPLE_QA_ITEM, incomplete_item]}
        with pytest.raises(ValidationError):
            RepairQABatchModel(**batch_data)


class TestCallLLMTypedBatch:
    """Test LLM batch call with Instructor"""

    def test_successful_batch_call(self):
        """Successful LLM call returns list of dicts"""
        mock_client = MagicMock()
        mock_response = MagicMock()
        
        # Create mock items
        mock_item_1 = MagicMock()
        mock_item_1.model_dump.return_value = SAMPLE_QA_ITEM
        mock_item_2 = MagicMock()
        mock_item_2.model_dump.return_value = SAMPLE_QA_ITEM
        
        mock_response.items = [mock_item_1, mock_item_2]
        mock_client.chat.completions.create.return_value = mock_response
        
        messages = [{"role": "user", "content": "Test"}]
        result = call_llm_typed_batch(mock_client, "test-model", messages)
        
        assert len(result) == 2
        assert all(isinstance(item, dict) for item in result)
        mock_client.chat.completions.create.assert_called_once()

    def test_batch_call_parameters(self):
        """LLM call should use correct parameters"""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_item = MagicMock()
        mock_item.model_dump.return_value = SAMPLE_QA_ITEM
        mock_response.items = [mock_item]
        mock_client.chat.completions.create.return_value = mock_response
        
        messages = [{"role": "user", "content": "Test"}]
        call_llm_typed_batch(mock_client, "test-model", messages)
        
        # Verify correct parameters were passed
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "test-model"
        assert call_args.kwargs["messages"] == messages
        assert call_args.kwargs["temperature"] == 0.7
        assert call_args.kwargs["response_model"] == RepairQABatchModel


class TestGenerateItemsFromTemplate:
    """Test template-based item generation"""

    def test_successful_generation(self):
        """Successful generation from template"""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_item = MagicMock()
        mock_item.model_dump.return_value = SAMPLE_QA_ITEM
        mock_response.items = [mock_item]
        mock_client.chat.completions.create.return_value = mock_response
        
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.jsonl"
            logger = JsonEventLogger(log_path=str(log_path), script_name="test", model="test")
            
            template_func = lambda count: [{"role": "user", "content": "test"}]
            result = generate_items_from_template(
                mock_client,
                "test-model",
                template_func,
                "plumbing",
                1,
                logger,
            )
            
            assert len(result) == 1
            assert result[0] == SAMPLE_QA_ITEM

    def test_generation_logs_count(self):
        """Generation should log validated count"""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_item = MagicMock()
        mock_item.model_dump.return_value = SAMPLE_QA_ITEM
        mock_response.items = [mock_item, mock_item]
        mock_client.chat.completions.create.return_value = mock_response
        
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.jsonl"
            logger = JsonEventLogger(log_path=str(log_path), script_name="test", model="test")
            
            template_func = lambda count: [{"role": "user", "content": "test"}]
            result = generate_items_from_template(
                mock_client,
                "test-model",
                template_func,
                "electrical",
                2,
                logger,
            )
            
            assert len(result) == 2
            
            # Check that log file contains validation message
            log_content = log_path.read_text()
            assert "validated" in log_content.lower()


class TestGenerateCandidateItems:
    """Test candidate item collection with retries"""

    def test_successful_collection_on_first_try(self):
        """Successful collection on first LLM call"""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_item = MagicMock()
        mock_item.model_dump.return_value = SAMPLE_QA_ITEM
        mock_response.items = [mock_item] * 5
        mock_client.chat.completions.create.return_value = mock_response
        
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.jsonl"
            logger = JsonEventLogger(log_path=str(log_path), script_name="test", model="test")
            
            template_func = lambda count: [{"role": "user", "content": "test"}]
            result = generate_candidate_items(
                mock_client,
                "test-model",
                template_func,
                "plumbing",
                5,
                logger,
            )
            
            assert len(result) == 5
            assert mock_client.chat.completions.create.call_count == 1

    def test_retry_on_error(self):
        """Should retry on error and eventually succeed"""
        mock_client = MagicMock()
        mock_item = MagicMock()
        mock_item.model_dump.return_value = SAMPLE_QA_ITEM
        
        # First call fails, second succeeds
        mock_response_success = MagicMock()
        mock_response_success.items = [mock_item] * 5
        
        mock_client.chat.completions.create.side_effect = [
            RuntimeError("API Error"),
            mock_response_success,
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.jsonl"
            logger = JsonEventLogger(log_path=str(log_path), script_name="test", model="test")
            
            template_func = lambda count: [{"role": "user", "content": "test"}]
            result = generate_candidate_items(
                mock_client,
                "test-model",
                template_func,
                "electrical",
                5,
                logger,
            )
            
            assert len(result) == 5
            assert mock_client.chat.completions.create.call_count == 2

    def test_exhausted_retries_raises_error(self):
        """Should raise error after exhausting retries"""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("API Error")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.jsonl"
            logger = JsonEventLogger(log_path=str(log_path), script_name="test", model="test")
            
            template_func = lambda count: [{"role": "user", "content": "test"}]
            with pytest.raises(RuntimeError):
                generate_candidate_items(
                    mock_client,
                    "test-model",
                    template_func,
                    "hvac",
                    5,
                    logger,
                )
            
            # Should have tried MAX_LLM_CALLS_PER_ATTEMPT times
            assert mock_client.chat.completions.create.call_count == MAX_LLM_CALLS_PER_ATTEMPT


class TestSaveRawDatasetAppend:
    """Test appending rows to JSONL file"""

    def test_append_to_empty_file(self):
        """Append rows to new empty file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.jsonl"
            rows = [SAMPLE_QA_ITEM, SAMPLE_QA_ITEM]
            
            save_raw_dataset_append(rows, str(output_path))
            
            # Verify file was created with correct content
            assert output_path.exists()
            lines = output_path.read_text().strip().split("\n")
            assert len(lines) == 2
            assert all(json.loads(line) == SAMPLE_QA_ITEM for line in lines)

    def test_append_to_existing_file(self):
        """Append rows to existing file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.jsonl"
            
            # Write initial rows
            save_raw_dataset_append([SAMPLE_QA_ITEM], str(output_path))
            
            # Append more rows
            save_raw_dataset_append([SAMPLE_QA_ITEM], str(output_path))
            
            lines = output_path.read_text().strip().split("\n")
            assert len(lines) == 2

    def test_save_empty_list(self):
        """Saving empty list should not create file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.jsonl"
            save_raw_dataset_append([], str(output_path))
            # File may not exist or be empty
            if output_path.exists():
                assert output_path.read_text() == ""

    def test_json_serialization(self):
        """Rows should be properly serialized as JSON"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.jsonl"
            save_raw_dataset_append([SAMPLE_QA_ITEM], str(output_path))
            
            lines = output_path.read_text().strip().split("\n")
            loaded = json.loads(lines[0])
            assert loaded == SAMPLE_QA_ITEM


class TestParseArgs:
    """Test command line argument parsing"""

    def test_default_args(self, monkeypatch):
        """Default arguments should use sensible defaults"""
        monkeypatch.setattr("sys.argv", ["dataset_generation.py"])
        args = parse_args()
        
        assert args.count == 10
        assert args.model == dataset_generation.DEFAULT_GROQ_MODEL
        assert "raw_diy_dataset" in args.raw_output

    def test_custom_count(self, monkeypatch):
        """Custom count argument"""
        monkeypatch.setattr("sys.argv", ["dataset_generation.py", "--count", "20"])
        args = parse_args()
        assert args.count == 20

    def test_custom_model(self, monkeypatch):
        """Custom model argument"""
        monkeypatch.setattr("sys.argv", ["dataset_generation.py", "--model", "custom-model"])
        args = parse_args()
        assert args.model == "custom-model"

    def test_custom_output_path(self, monkeypatch):
        """Custom output path argument"""
        monkeypatch.setattr("sys.argv", ["dataset_generation.py", "--raw-output", "/custom/path.jsonl"])
        args = parse_args()
        assert args.raw_output == "/custom/path.jsonl"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
