import json
import os
import tempfile
import unittest

from sft.collator import DataCollatorForChatSFT
from sft.data_utils import load_sft_records, split_sft_records, validate_messages_format
from sft.flow_validation import validate_local_sft_config, validate_production_sft_config
from sft.toy_tokenizer import ToyChatTokenizer


SAMPLES = [
    {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ],
        "source": "test",
    },
    {
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Two plus two?"},
            {"role": "assistant", "content": "Four."},
        ],
        "source": "test",
    },
    {
        "messages": [
            {"role": "user", "content": "Say done."},
            {"role": "assistant", "content": "done"},
        ],
        "source": "test",
    },
]


class SftDataUtilsTest(unittest.TestCase):
    def test_load_and_validate_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                for row in SAMPLES:
                    f.write(json.dumps(row) + "\n")

            records = load_sft_records(path)

        self.assertEqual(len(records), len(SAMPLES))
        validate_messages_format(records)

    def test_invalid_row_without_assistant_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bad.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"messages": [{"role": "user", "content": "hi"}]}) + "\n")

            with self.assertRaises(ValueError):
                load_sft_records(path)

    def test_split_records_keeps_train_and_val(self):
        train, val = split_sft_records(load_records_from_samples(), val_fraction=0.34, seed=1)
        self.assertEqual(len(train) + len(val), len(SAMPLES))
        self.assertGreaterEqual(len(val), 1)


class SftCollatorTest(unittest.TestCase):
    def test_masks_user_tokens_and_keeps_assistant_tokens(self):
        tokenizer = ToyChatTokenizer(vocab_size=128)
        collator = DataCollatorForChatSFT(tokenizer=tokenizer, max_seq_length=64)
        batch = collator([
            {
                "messages": [
                    {"role": "user", "content": "U"},
                    {"role": "assistant", "content": "A"},
                ]
            }
        ])

        labels = batch["labels"][0].tolist()
        assistant_token = tokenizer.encode("A", add_special_tokens=False)[0]
        user_token = tokenizer.encode("U", add_special_tokens=False)[0]

        self.assertIn(assistant_token, labels)
        self.assertIn(tokenizer.im_end_id, labels)
        self.assertNotIn(user_token, [x for x in labels if x != -100])
        self.assertEqual(batch["input_ids"].shape, batch["labels"].shape)


class SftFlowValidationTest(unittest.TestCase):
    def test_local_sft_config_passes(self):
        train_cfg = {
            "use_cpu": True,
            "tokenizer_type": "toy_chatml",
            "data_dir": "./data/local_smoke/sft",
            "output_dir": "./artifacts/local/sft/checkpoints",
            "pretrained_model_path": "./artifacts/local/sft/pretrained",
            "resume_from_last_checkpoint": False,
        }
        validate_local_sft_config(train_cfg, {"vocab_size": 256})

    def test_production_sft_config_passes(self):
        train_cfg = {
            "use_cpu": False,
            "tokenizer_name": "Qwen/Qwen3.5-0.8B",
            "deepspeed": "configs/production/deepspeed_zero2.json",
            "data_dir": "./data/production/sft/zh_first_v0_3b",
            "output_dir": "./artifacts/production/sft/checkpoints",
            "pretrained_model_path": "./artifacts/production/checkpoints_zh_first_3b/final",
            "resume_from_last_checkpoint": True,
        }
        validate_production_sft_config(train_cfg, {"vocab_size": 248320})

    def test_production_sft_blocks_local_data_path(self):
        train_cfg = {
            "use_cpu": False,
            "tokenizer_name": "Qwen/Qwen3.5-0.8B",
            "deepspeed": "configs/production/deepspeed_zero2.json",
            "data_dir": "./data/local_smoke/sft",
            "output_dir": "./artifacts/production/sft/checkpoints",
            "pretrained_model_path": "./artifacts/production/checkpoints_zh_first_3b/final",
            "resume_from_last_checkpoint": True,
        }
        with self.assertRaises(ValueError):
            validate_production_sft_config(train_cfg, {"vocab_size": 248320})


def load_records_from_samples():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "data.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for row in SAMPLES:
                f.write(json.dumps(row) + "\n")
        return load_sft_records(path)


if __name__ == "__main__":
    unittest.main()

