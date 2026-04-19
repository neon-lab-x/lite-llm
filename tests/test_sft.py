"""Unit tests for SFT flow: data loading, format validation, loss masking, flow isolation."""

import json
import os
import tempfile
import unittest

from lite_llm.sft_data_utils import (
    load_sft_dataset,
    split_sft_train_val,
    validate_messages_format,
)


SAMPLE_CONVERSATIONS = [
    {"messages": [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]},
    {"messages": [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."},
    ]},
    {"messages": [
        {"role": "user", "content": "Goodbye"},
        {"role": "assistant", "content": "See you!"},
    ]},
    {"messages": [
        {"role": "user", "content": "Thanks"},
        {"role": "assistant", "content": "You're welcome."},
    ]},
    {"messages": [
        {"role": "user", "content": "What color is the sky?"},
        {"role": "assistant", "content": "The sky is blue."},
    ]},
]


class TestSFTDataLoading(unittest.TestCase):
    def test_load_single_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.jsonl")
            with open(path, "w") as f:
                for conv in SAMPLE_CONVERSATIONS:
                    f.write(json.dumps(conv) + "\n")

            ds = load_sft_dataset(path)
            self.assertEqual(len(ds), len(SAMPLE_CONVERSATIONS))
            self.assertIn("messages", ds.column_names)

    def test_load_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, conv in enumerate(SAMPLE_CONVERSATIONS[:2]):
                path = os.path.join(tmpdir, f"part{i}.jsonl")
                with open(path, "w") as f:
                    f.write(json.dumps(conv) + "\n")

            ds = load_sft_dataset(tmpdir)
            self.assertEqual(len(ds), 2)

    def test_load_empty_directory_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                load_sft_dataset(tmpdir)


class TestSFTDataSplit(unittest.TestCase):
    def test_split_produces_correct_sizes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.jsonl")
            with open(path, "w") as f:
                for conv in SAMPLE_CONVERSATIONS:
                    f.write(json.dumps(conv) + "\n")

            ds = load_sft_dataset(path)
            train_ds, val_ds = split_sft_train_val(ds, val_fraction=0.2, seed=42)
            self.assertEqual(len(train_ds) + len(val_ds), len(ds))
            self.assertGreater(len(val_ds), 0)

    def test_zero_fraction_returns_none_val(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.jsonl")
            with open(path, "w") as f:
                for conv in SAMPLE_CONVERSATIONS:
                    f.write(json.dumps(conv) + "\n")

            ds = load_sft_dataset(path)
            train_ds, val_ds = split_sft_train_val(ds, val_fraction=0.0)
            self.assertEqual(len(train_ds), len(ds))
            self.assertIsNone(val_ds)


class TestValidateMessagesFormat(unittest.TestCase):
    def test_valid_format_passes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.jsonl")
            with open(path, "w") as f:
                for conv in SAMPLE_CONVERSATIONS:
                    f.write(json.dumps(conv) + "\n")

            ds = load_sft_dataset(path)
            validate_messages_format(ds)  # should not raise

    def test_missing_messages_column_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.jsonl")
            with open(path, "w") as f:
                f.write(json.dumps({"text": "hello"}) + "\n")

            ds = load_sft_dataset(path)
            with self.assertRaises(ValueError) as ctx:
                validate_messages_format(ds)
            self.assertIn("messages", str(ctx.exception))

    def test_missing_role_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.jsonl")
            with open(path, "w") as f:
                f.write(json.dumps({"messages": [{"content": "hi"}]}) + "\n")

            ds = load_sft_dataset(path)
            with self.assertRaises(ValueError) as ctx:
                validate_messages_format(ds)
            self.assertIn("role", str(ctx.exception))


class TestSFTFlowValidation(unittest.TestCase):
    """Test SFT flow validators enforce isolation rules."""

    def test_local_sft_config_passes(self):
        from lite_llm.flow_validation import validate_local_sft_config

        train_cfg = {
            "use_cpu": True,
            "tokenizer_name": "Qwen/Qwen3.5-0.8B",
            "data_dir": "./data/local_smoke/sft",
            "output_dir": "./artifacts/local/sft_checkpoints",
            "pretrained_model_path": "./artifacts/local/sft_pretrained",
            "resume_from_last_checkpoint": False,
        }
        model_cfg = {"vocab_size": 248320}
        validate_local_sft_config(train_cfg, model_cfg)  # should not raise

    def test_local_sft_requires_tokenizer(self):
        from lite_llm.flow_validation import validate_local_sft_config

        train_cfg = {
            "use_cpu": True,
            "data_dir": "./data/local_smoke/sft",
            "output_dir": "./artifacts/local/sft_checkpoints",
            "pretrained_model_path": "./artifacts/local/sft_pretrained",
            "resume_from_last_checkpoint": False,
        }
        model_cfg = {"vocab_size": 248320}
        with self.assertRaises(ValueError):
            validate_local_sft_config(train_cfg, model_cfg)

    def test_local_sft_requires_pretrained_path(self):
        from lite_llm.flow_validation import validate_local_sft_config

        train_cfg = {
            "use_cpu": True,
            "tokenizer_name": "Qwen/Qwen3.5-0.8B",
            "data_dir": "./data/local_smoke/sft",
            "output_dir": "./artifacts/local/sft_checkpoints",
            "resume_from_last_checkpoint": False,
        }
        model_cfg = {"vocab_size": 248320}
        with self.assertRaises(ValueError):
            validate_local_sft_config(train_cfg, model_cfg)

    def test_local_sft_blocks_deepspeed(self):
        from lite_llm.flow_validation import validate_local_sft_config

        train_cfg = {
            "use_cpu": True,
            "tokenizer_name": "Qwen/Qwen3.5-0.8B",
            "data_dir": "./data/local_smoke/sft",
            "output_dir": "./artifacts/local/sft_checkpoints",
            "pretrained_model_path": "./artifacts/local/sft_pretrained",
            "resume_from_last_checkpoint": False,
            "deepspeed": "some_config.json",
        }
        model_cfg = {"vocab_size": 248320}
        with self.assertRaises(ValueError):
            validate_local_sft_config(train_cfg, model_cfg)

    def test_production_sft_config_passes(self):
        from lite_llm.flow_validation import validate_production_sft_config

        train_cfg = {
            "use_cpu": False,
            "tokenizer_name": "Qwen/Qwen3.5-0.8B",
            "deepspeed": "configs/production/deepspeed_zero2.json",
            "data_dir": "./data/production/sft",
            "output_dir": "./artifacts/production/sft_checkpoints",
            "pretrained_model_path": "./artifacts/production/checkpoints/final",
            "resume_from_last_checkpoint": True,
        }
        model_cfg = {"vocab_size": 248320}
        validate_production_sft_config(train_cfg, model_cfg)  # should not raise

    def test_production_sft_requires_deepspeed(self):
        from lite_llm.flow_validation import validate_production_sft_config

        train_cfg = {
            "use_cpu": False,
            "tokenizer_name": "Qwen/Qwen3.5-0.8B",
            "data_dir": "./data/production/sft",
            "output_dir": "./artifacts/production/sft_checkpoints",
            "pretrained_model_path": "./artifacts/production/checkpoints/final",
            "resume_from_last_checkpoint": True,
        }
        model_cfg = {"vocab_size": 248320}
        with self.assertRaises(ValueError):
            validate_production_sft_config(train_cfg, model_cfg)

    def test_production_sft_blocks_wrong_paths(self):
        from lite_llm.flow_validation import validate_production_sft_config

        train_cfg = {
            "use_cpu": False,
            "tokenizer_name": "Qwen/Qwen3.5-0.8B",
            "deepspeed": "configs/production/deepspeed_zero2.json",
            "data_dir": "./data/local_smoke/sft",  # wrong prefix
            "output_dir": "./artifacts/production/sft_checkpoints",
            "pretrained_model_path": "./artifacts/production/checkpoints/final",
            "resume_from_last_checkpoint": True,
        }
        model_cfg = {"vocab_size": 248320}
        with self.assertRaises(ValueError):
            validate_production_sft_config(train_cfg, model_cfg)

    def test_production_sft_blocks_prefix_bypass_paths(self):
        from lite_llm.flow_validation import validate_production_sft_config

        train_cfg = {
            "use_cpu": False,
            "tokenizer_name": "Qwen/Qwen3.5-0.8B",
            "deepspeed": "configs/production/deepspeed_zero2.json",
            "data_dir": "./data/production_evil/sft",  # prefix-bypass attempt
            "output_dir": "./artifacts/production/sft_checkpoints",
            "pretrained_model_path": "./artifacts/production/checkpoints/final",
            "resume_from_last_checkpoint": True,
        }
        model_cfg = {"vocab_size": 248320}
        with self.assertRaises(ValueError):
            validate_production_sft_config(train_cfg, model_cfg)


class TestLocalSFTPrepareData(unittest.TestCase):
    def test_prepare_creates_valid_jsonl(self):
        """Write SAMPLE_CONVERSATIONS as JSONL and verify it round-trips correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = os.path.join(tmpdir, "sft_data.jsonl")
            with open(jsonl_path, "w") as f:
                for conv in SAMPLE_CONVERSATIONS:
                    f.write(json.dumps(conv) + "\n")

            ds = load_sft_dataset(jsonl_path)
            self.assertEqual(len(ds), len(SAMPLE_CONVERSATIONS))
            validate_messages_format(ds)  # should not raise


class TestValidateMessagesFormatEdgeCases(unittest.TestCase):
    def test_bad_row_in_middle_raises(self):
        """validate_messages_format should catch a corrupt row in the middle."""
        convs = list(SAMPLE_CONVERSATIONS)
        # Insert a corrupt entry at position 2
        convs.insert(2, {"messages": [{"content": "no role here"}]})
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.jsonl")
            with open(path, "w") as f:
                for conv in convs:
                    f.write(json.dumps(conv) + "\n")
            ds = load_sft_dataset(path)
            with self.assertRaises(ValueError):
                validate_messages_format(ds)

    def test_empty_messages_list_raises(self):
        """validate_messages_format should reject an empty messages list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.jsonl")
            with open(path, "w") as f:
                f.write(json.dumps({"messages": []}) + "\n")
            ds = load_sft_dataset(path)
            with self.assertRaises(ValueError):
                validate_messages_format(ds)

    def test_invalid_role_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.jsonl")
            bad = {"messages": [{"role": "bot", "content": "hello"}]}
            with open(path, "w") as f:
                f.write(json.dumps(bad) + "\n")
            ds = load_sft_dataset(path)
            with self.assertRaises(ValueError) as ctx:
                validate_messages_format(ds)
            self.assertIn("invalid role", str(ctx.exception))

    def test_empty_content_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.jsonl")
            bad = {"messages": [{"role": "assistant", "content": "   "}]}
            with open(path, "w") as f:
                f.write(json.dumps(bad) + "\n")
            ds = load_sft_dataset(path)
            with self.assertRaises(ValueError) as ctx:
                validate_messages_format(ds)
            self.assertIn("non-empty string", str(ctx.exception))

    def test_bad_row_after_50_still_raises_by_default(self):
        """Default validation should scan beyond the first 50 rows."""
        convs = list(SAMPLE_CONVERSATIONS) * 12  # 60 rows
        convs[55] = {"messages": [{"content": "missing role in late row"}]}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.jsonl")
            with open(path, "w") as f:
                for conv in convs:
                    f.write(json.dumps(conv) + "\n")
            ds = load_sft_dataset(path)
            with self.assertRaises(ValueError):
                validate_messages_format(ds)


class TestSplitSFTTrainValEdgeCases(unittest.TestCase):
    def test_val_fraction_ge_one_raises(self):
        """val_fraction >= 1.0 must raise a clear error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.jsonl")
            with open(path, "w") as f:
                for conv in SAMPLE_CONVERSATIONS:
                    f.write(json.dumps(conv) + "\n")
            ds = load_sft_dataset(path)
            with self.assertRaises(ValueError):
                split_sft_train_val(ds, val_fraction=1.0)

    def test_val_fraction_gt_one_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.jsonl")
            with open(path, "w") as f:
                for conv in SAMPLE_CONVERSATIONS:
                    f.write(json.dumps(conv) + "\n")
            ds = load_sft_dataset(path)
            with self.assertRaises(ValueError):
                split_sft_train_val(ds, val_fraction=1.5)


class TestLoadSFTDatasetEdgeCases(unittest.TestCase):
    def test_nonexistent_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_sft_dataset("/nonexistent/path/data.jsonl")

    def test_directory_with_non_jsonl_files_ignored(self):
        """Non-JSONL files inside a directory should be ignored, not loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write one valid JSONL and one unrelated file
            jsonl_path = os.path.join(tmpdir, "data.jsonl")
            with open(jsonl_path, "w") as f:
                f.write(json.dumps(SAMPLE_CONVERSATIONS[0]) + "\n")
            txt_path = os.path.join(tmpdir, "readme.txt")
            with open(txt_path, "w") as f:
                f.write("not a data file\n")

            ds = load_sft_dataset(tmpdir)
            self.assertEqual(len(ds), 1)


class TestShareGPTConverter(unittest.TestCase):
    """Unit tests for the ShareGPT → ChatML converter in sft_prepare_data."""

    def _get_converter(self):
        import importlib.util
        import sys
        spec = importlib.util.spec_from_file_location(
            "sft_prepare_data",
            os.path.join(
                os.path.dirname(__file__),
                "..", "scripts", "production", "sft_prepare_data.py",
            ),
        )
        module = importlib.util.module_from_spec(spec)
        # Stub out the 'datasets' import so we don't need the production extra
        sys.modules.setdefault("datasets", type(sys)("datasets"))
        sys.modules.setdefault("yaml", __import__("yaml"))
        spec.loader.exec_module(module)
        return module._convert_sharegpt

    def test_basic_conversion(self):
        convert = self._get_converter()
        example = {
            "conversations": [
                {"from": "human", "value": "Hello"},
                {"from": "gpt", "value": "Hi there!"},
            ]
        }
        result = convert(example)
        self.assertIsNotNone(result)
        msgs = result["messages"]
        self.assertEqual(msgs[0], {"role": "user", "content": "Hello"})
        self.assertEqual(msgs[1], {"role": "assistant", "content": "Hi there!"})

    def test_system_turn(self):
        convert = self._get_converter()
        example = {
            "conversations": [
                {"from": "system", "value": "Be helpful."},
                {"from": "human", "value": "Hi"},
                {"from": "gpt", "value": "Hello!"},
            ]
        }
        result = convert(example)
        self.assertEqual(result["messages"][0]["role"], "system")

    def test_empty_conversations_returns_none(self):
        convert = self._get_converter()
        self.assertIsNone(convert({"conversations": []}))

    def test_missing_conversations_returns_none(self):
        convert = self._get_converter()
        self.assertIsNone(convert({}))


class TestSFTRunnerImport(unittest.TestCase):
    def test_sft_runner_module_imports(self):
        """Catch TRL API drift that breaks the SFT training entrypoint import."""
        import importlib

        module = importlib.import_module("lite_llm.sft_runner")
        self.assertTrue(hasattr(module, "run_sft_training"))


class TestAssistantMaskingHelpers(unittest.TestCase):
    def test_mask_assistant_spans_multiple_turns(self):
        from lite_llm.sft_runner import _mask_assistant_spans

        input_ids = [1, 2, 10, 11, 20, 21, 99, 3, 4, 10, 11, 30, 99]
        response_ids = [10, 11]
        end_ids = [99]
        labels = _mask_assistant_spans(input_ids, response_ids, end_ids)
        expected = [-100, -100, -100, -100, 20, 21, -100, -100, -100, -100, -100, 30, -100]
        self.assertEqual(labels, expected)

    def test_mask_assistant_spans_no_assistant_marker(self):
        from lite_llm.sft_runner import _mask_assistant_spans

        input_ids = [1, 2, 3, 4]
        labels = _mask_assistant_spans(input_ids, response_token_ids=[10], end_token_ids=[99])
        self.assertEqual(labels, [-100, -100, -100, -100])

    def test_mask_assistant_spans_truncated_turn_excludes_padding(self):
        from lite_llm.sft_runner import _mask_assistant_spans

        # [10, 11] is assistant marker, no end marker (truncated assistant turn).
        # Trailing zeros are right padding and must stay masked.
        input_ids = [1, 10, 11, 20, 0, 0]
        attention_mask = [1, 1, 1, 1, 0, 0]
        labels = _mask_assistant_spans(
            input_ids,
            response_token_ids=[10, 11],
            end_token_ids=[99],
            attention_mask=attention_mask,
        )
        self.assertEqual(labels, [-100, -100, -100, 20, -100, -100])


class TestSFTRunnerCompatHelpers(unittest.TestCase):
    def test_build_sft_processing_kwargs_prefers_processing_class(self):
        from lite_llm.sft_runner import _build_sft_processing_kwargs

        tokenizer = object()
        kwargs = _build_sft_processing_kwargs(
            tokenizer=tokenizer,
            sft_init_params={"processing_class": object(), "tokenizer": object()},
        )
        self.assertEqual(kwargs, {"processing_class": tokenizer})

    def test_build_sft_processing_kwargs_falls_back_to_tokenizer(self):
        from lite_llm.sft_runner import _build_sft_processing_kwargs

        tokenizer = object()
        kwargs = _build_sft_processing_kwargs(
            tokenizer=tokenizer,
            sft_init_params={"tokenizer": object()},
        )
        self.assertEqual(kwargs, {"tokenizer": tokenizer})

    def test_build_sft_processing_kwargs_raises_on_unknown_signature(self):
        from lite_llm.sft_runner import _build_sft_processing_kwargs

        with self.assertRaises(TypeError):
            _build_sft_processing_kwargs(tokenizer=object(), sft_init_params={})


if __name__ == "__main__":
    unittest.main()
