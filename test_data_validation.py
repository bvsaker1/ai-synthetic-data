"""
Unit tests for data_validation.py

Tests the deduplication pipeline including:
- Text normalization for semantic matching
- Embedding generation and similarity computation
- Duplicate detection using vector similarity
- Invalid row handling and logging
- Pydantic model validation
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pydantic import ValidationError

import data_validation
from data_validation import (
    DEFAULT_DEDUP_THRESHOLD,
    EMBEDDING_MODEL,
    RepairQAModel,
    _build_embedding_text,
    _normalize_text,
    deduplicate_dataset,
    parse_args,
)
from logging_utils import JsonEventLogger


# Sample valid QA item
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
    "judge_scoring": "",
}

# Semantically similar variant
SIMILAR_QA_ITEM = {
    **SAMPLE_QA_ITEM,
    "id": "qa_2",
    "question": "How do I repair a leaky faucet?",  # Slight variation
    "answer": "Turn off water and swap the washers.",  # Slight variation
}

# Completely different item
DIFFERENT_QA_ITEM = {
    **SAMPLE_QA_ITEM,
    "id": "qa_3",
    "category": "electrical",
    "question": "How do I replace a light switch?",
    "answer": "Turn off the breaker and swap the switch.",
    "equipment_problem": "broken light switch",
}


class TestNormalizeText:
    """Test text normalization for deduplication"""

    def test_lowercase_conversion(self):
        """Text should be converted to lowercase"""
        result = _normalize_text("HELLO World")
        assert result == "hello world"

    def test_whitespace_collapse(self):
        """Multiple spaces should collapse to single space"""
        result = _normalize_text("Hello    World")
        assert result == "hello world"
        
        result = _normalize_text("  Hello  \n  World  ")
        assert result == "hello world"

    def test_strip_leading_trailing(self):
        """Leading/trailing whitespace should be removed"""
        result = _normalize_text("  hello  ")
        assert result == "hello"

    def test_empty_string(self):
        """Empty string should return empty string"""
        result = _normalize_text("")
        assert result == ""

    def test_special_characters(self):
        """Special characters should be preserved"""
        result = _normalize_text("hello-world!")
        assert result == "hello-world!"

    def test_numeric_conversion(self):
        """Numeric values should be converted to string"""
        result = _normalize_text(123)
        assert result == "123"


class TestBuildEmbeddingText:
    """Test embedding text construction"""

    def test_combines_key_fields(self):
        """Embedding text should combine question, equipment_problem, answer"""
        text = _build_embedding_text(SAMPLE_QA_ITEM)
        assert "leaky kitchen faucet" in text.lower()
        assert "leaky faucet" in text.lower() or "faucet" in text.lower()
        assert "replace" in text.lower() or "washers" in text.lower()

    def test_missing_fields_handled(self):
        """Missing fields should not raise errors"""
        item = {"question": "test"}
        text = _build_embedding_text(item)
        assert isinstance(text, str)
        assert "test" in text.lower()

    def test_normalization_applied(self):
        """Embedding text should be normalized"""
        item = {
            "question": "  HELLO  World  ",
            "equipment_problem": "TEST",
            "answer": "  ANSWER  ",
        }
        text = _build_embedding_text(item)
        assert text == text.lower()  # Should be lowercase


class TestRepairQAModel:
    """Test Pydantic model validation"""

    def test_valid_item(self):
        """Valid item should pass validation"""
        item = RepairQAModel(**SAMPLE_QA_ITEM)
        assert item.id == "qa_1"
        assert item.category == "plumbing"

    def test_missing_required_field(self):
        """Missing required field should raise ValidationError"""
        incomplete = SAMPLE_QA_ITEM.copy()
        del incomplete["id"]
        with pytest.raises(ValidationError):
            RepairQAModel(**incomplete)

    def test_empty_required_field(self):
        """Empty required field should raise ValidationError"""
        invalid = SAMPLE_QA_ITEM.copy()
        invalid["question"] = ""
        with pytest.raises(ValidationError):
            RepairQAModel(**invalid)

    def test_steps_minimum_length(self):
        """Steps must have at least 3 items"""
        invalid = SAMPLE_QA_ITEM.copy()
        invalid["steps"] = ["Step 1", "Step 2"]
        with pytest.raises(ValidationError):
            RepairQAModel(**invalid)

    def test_judge_scoring_default(self):
        """judge_scoring should default to empty string"""
        item = SAMPLE_QA_ITEM.copy()
        del item["judge_scoring"]
        validated = RepairQAModel(**item)
        assert validated.judge_scoring == ""


class TestDeduplicateDataset:
    """Test the deduplication pipeline"""

    def test_exact_duplicates_detected(self):
        """Exact duplicates should be flagged"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.jsonl"
            valid_path = Path(tmpdir) / "valid.jsonl"
            invalid_path = Path(tmpdir) / "invalid.jsonl"
            log_path = Path(tmpdir) / "log.jsonl"
            
            # Write input: exact duplicate
            input_path.write_text(
                json.dumps(SAMPLE_QA_ITEM) + "\n" +
                json.dumps(SAMPLE_QA_ITEM) + "\n"
            )
            
            logger = JsonEventLogger(log_path=str(log_path), script_name="test", model="test")
            
            with patch("data_validation.SentenceTransformer") as mock_transformer:
                mock_instance = MagicMock()
                mock_transformer.return_value = mock_instance
                
                # Both items get same embedding (exact match)
                embedding = np.array([1.0, 0.0, 0.0])
                mock_instance.encode.return_value = embedding
                
                unique, duplicates, invalid = deduplicate_dataset(
                    input_path=str(input_path),
                    valid_output_path=str(valid_path),
                    invalid_output_path=str(invalid_path),
                    rerun_model_validate=False,
                    dedup_threshold=0.95,
                    logger=logger,
                )
                
                assert unique == 1
                assert duplicates == 1
                assert invalid == 0
                
                # Check valid file has 1 item
                valid_lines = valid_path.read_text().strip().split("\n")
                assert len(valid_lines) == 1
                
                # Check invalid file has 1 item
                invalid_lines = invalid_path.read_text().strip().split("\n")
                assert len(invalid_lines) == 1
                invalid_item = json.loads(invalid_lines[0])
                assert invalid_item["reason"] == "semantic_duplicate"

    def test_diverse_items_not_deduplicated(self):
        """Diverse items should not be flagged as duplicates"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.jsonl"
            valid_path = Path(tmpdir) / "valid.jsonl"
            invalid_path = Path(tmpdir) / "invalid.jsonl"
            log_path = Path(tmpdir) / "log.jsonl"
            
            # Write input: different items
            input_path.write_text(
                json.dumps(SAMPLE_QA_ITEM) + "\n" +
                json.dumps(DIFFERENT_QA_ITEM) + "\n"
            )
            
            logger = JsonEventLogger(log_path=str(log_path), script_name="test", model="test")
            
            with patch("data_validation.SentenceTransformer") as mock_transformer:
                mock_instance = MagicMock()
                mock_transformer.return_value = mock_instance
                
                # Different embeddings (low similarity)
                embedding1 = np.array([1.0, 0.0, 0.0])
                embedding2 = np.array([0.0, 1.0, 0.0])
                mock_instance.encode.side_effect = [embedding1, embedding2]
                
                unique, duplicates, invalid = deduplicate_dataset(
                    input_path=str(input_path),
                    valid_output_path=str(valid_path),
                    invalid_output_path=str(invalid_path),
                    rerun_model_validate=False,
                    dedup_threshold=0.95,
                    logger=logger,
                )
                
                assert unique == 2
                assert duplicates == 0
                assert invalid == 0

    def test_invalid_json_handling(self):
        """Invalid JSON should be captured in invalid output"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.jsonl"
            valid_path = Path(tmpdir) / "valid.jsonl"
            invalid_path = Path(tmpdir) / "invalid.jsonl"
            log_path = Path(tmpdir) / "log.jsonl"
            
            # Write input: valid + invalid JSON
            input_path.write_text(
                json.dumps(SAMPLE_QA_ITEM) + "\n" +
                "{ invalid json\n"
            )
            
            logger = JsonEventLogger(log_path=str(log_path), script_name="test", model="test")
            
            with patch("data_validation.SentenceTransformer"):
                unique, duplicates, invalid = deduplicate_dataset(
                    input_path=str(input_path),
                    valid_output_path=str(valid_path),
                    invalid_output_path=str(invalid_path),
                    rerun_model_validate=False,
                    dedup_threshold=0.95,
                    logger=logger,
                )
                
                assert unique == 1
                assert invalid == 1
                
                invalid_lines = invalid_path.read_text().strip().split("\n")
                invalid_row = json.loads(invalid_lines[0])
                assert "error" in invalid_row

    def test_pydantic_validation_error_handling(self):
        """Pydantic validation errors should be captured"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.jsonl"
            valid_path = Path(tmpdir) / "valid.jsonl"
            invalid_path = Path(tmpdir) / "invalid.jsonl"
            log_path = Path(tmpdir) / "log.jsonl"
            
            # Write input: valid + missing required field
            invalid_item = SAMPLE_QA_ITEM.copy()
            del invalid_item["question"]
            
            input_path.write_text(
                json.dumps(SAMPLE_QA_ITEM) + "\n" +
                json.dumps(invalid_item) + "\n"
            )
            
            logger = JsonEventLogger(log_path=str(log_path), script_name="test", model="test")
            
            with patch("data_validation.SentenceTransformer"):
                unique, duplicates, invalid = deduplicate_dataset(
                    input_path=str(input_path),
                    valid_output_path=str(valid_path),
                    invalid_output_path=str(invalid_path),
                    rerun_model_validate=True,  # Enable validation
                    dedup_threshold=0.95,
                    logger=logger,
                )
                
                assert unique == 1
                assert invalid == 1

    def test_threshold_parameter_respected(self):
        """Different thresholds should affect duplicate detection"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.jsonl"
            valid_path = Path(tmpdir) / "valid.jsonl"
            invalid_path = Path(tmpdir) / "invalid.jsonl"
            log_path = Path(tmpdir) / "log.jsonl"
            
            # Write input: 2 similar items
            input_path.write_text(
                json.dumps(SAMPLE_QA_ITEM) + "\n" +
                json.dumps(SIMILAR_QA_ITEM) + "\n"
            )
            
            logger = JsonEventLogger(log_path=str(log_path), script_name="test", model="test")
            
            with patch("data_validation.SentenceTransformer") as mock_transformer:
                mock_instance = MagicMock()
                mock_transformer.return_value = mock_instance
                
                # Create embeddings with true ~0.85 cosine similarity
                # Using unit vectors: similarity = dot product / (||a|| * ||b||)
                embedding1 = np.array([1.0, 0.0])
                embedding2 = np.array([0.85, 0.527])  # This gives ~0.85 similarity when normalized
                mock_instance.encode.side_effect = [embedding1, embedding2]
                
                # With threshold 0.95, similarity 0.85 should NOT trigger dedup
                unique, duplicates, invalid = deduplicate_dataset(
                    input_path=str(input_path),
                    valid_output_path=str(valid_path),
                    invalid_output_path=str(invalid_path),
                    rerun_model_validate=False,
                    dedup_threshold=0.95,
                    logger=logger,
                )
                
                # Items should NOT be flagged as duplicates
                assert unique == 2
                assert duplicates == 0

    def test_empty_lines_skipped(self):
        """Empty lines in input should be skipped"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.jsonl"
            valid_path = Path(tmpdir) / "valid.jsonl"
            invalid_path = Path(tmpdir) / "invalid.jsonl"
            log_path = Path(tmpdir) / "log.jsonl"
            
            # Write input with empty lines
            input_path.write_text(
                json.dumps(SAMPLE_QA_ITEM) + "\n" +
                "\n" +  # Empty line
                json.dumps(DIFFERENT_QA_ITEM) + "\n"
            )
            
            logger = JsonEventLogger(log_path=str(log_path), script_name="test", model="test")
            
            with patch("data_validation.SentenceTransformer") as mock_transformer:
                mock_instance = MagicMock()
                mock_transformer.return_value = mock_instance
                
                # Two distinct embeddings for the two valid items
                embedding1 = np.array([1.0, 0.0, 0.0])
                embedding2 = np.array([0.0, 1.0, 0.0])
                mock_instance.encode.side_effect = [embedding1, embedding2]
                
                unique, duplicates, invalid = deduplicate_dataset(
                    input_path=str(input_path),
                    valid_output_path=str(valid_path),
                    invalid_output_path=str(invalid_path),
                    rerun_model_validate=False,
                    dedup_threshold=0.95,
                    logger=logger,
                )
                
                # Empty lines should be skipped, only 2 valid items
                assert unique == 2
                assert invalid == 0

    def test_similarity_score_recorded(self):
        """Duplicate records should include similarity score"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.jsonl"
            valid_path = Path(tmpdir) / "valid.jsonl"
            invalid_path = Path(tmpdir) / "invalid.jsonl"
            log_path = Path(tmpdir) / "log.jsonl"
            
            input_path.write_text(
                json.dumps(SAMPLE_QA_ITEM) + "\n" +
                json.dumps(SIMILAR_QA_ITEM) + "\n"
            )
            
            logger = JsonEventLogger(log_path=str(log_path), script_name="test", model="test")
            
            with patch("data_validation.SentenceTransformer") as mock_transformer:
                mock_instance = MagicMock()
                mock_transformer.return_value = mock_instance
                
                embedding1 = np.array([1.0, 0.0, 0.0])
                embedding2 = np.array([0.99, 0.01, 0.0])  # 99% similar
                mock_instance.encode.side_effect = [embedding1, embedding2]
                
                unique, duplicates, invalid = deduplicate_dataset(
                    input_path=str(input_path),
                    valid_output_path=str(valid_path),
                    invalid_output_path=str(invalid_path),
                    rerun_model_validate=False,
                    dedup_threshold=0.95,
                    logger=logger,
                )
                
                invalid_lines = invalid_path.read_text().strip().split("\n")
                duplicate_record = json.loads(invalid_lines[0])
                assert "similarity" in duplicate_record
                assert duplicate_record["similarity"] >= 0.95


class TestParseArgs:
    """Test command line argument parsing"""

    def test_default_args(self, monkeypatch):
        """Default arguments should use sensible defaults"""
        monkeypatch.setattr("sys.argv", ["data_validation.py"])
        args = parse_args()
        
        assert args.dedup_threshold == DEFAULT_DEDUP_THRESHOLD
        assert args.rerun_model_validate == False
        assert "raw_diy_dataset" in args.input
        assert "diy_dataset" in args.valid_output
        assert "invalid_diy_dataset" in args.invalid_output

    def test_custom_threshold(self, monkeypatch):
        """Custom threshold argument"""
        monkeypatch.setattr("sys.argv", ["data_validation.py", "--dedup-threshold", "0.85"])
        args = parse_args()
        assert args.dedup_threshold == 0.85

    def test_rerun_model_validate_flag(self, monkeypatch):
        """Rerun model validate flag"""
        monkeypatch.setattr("sys.argv", ["data_validation.py", "--rerun-model-validate"])
        args = parse_args()
        assert args.rerun_model_validate == True

    def test_custom_paths(self, monkeypatch):
        """Custom path arguments"""
        monkeypatch.setattr("sys.argv", [
            "data_validation.py",
            "--input", "/custom/input.jsonl",
            "--valid-output", "/custom/valid.jsonl",
            "--invalid-output", "/custom/invalid.jsonl",
        ])
        args = parse_args()
        assert args.input == "/custom/input.jsonl"
        assert args.valid_output == "/custom/valid.jsonl"
        assert args.invalid_output == "/custom/invalid.jsonl"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
