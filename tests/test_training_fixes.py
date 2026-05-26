import os
import importlib.util
import tempfile
import unittest
from unittest.mock import MagicMock

import numpy as np
import torch
import yaml

from lite_llm.configuration import LiteLlmConfig
from lite_llm.data_utils import (
    DataCollatorForPretraining,
    load_tokenized_dataset,
    split_train_val,
    tokenize_and_save,
)
from lite_llm.flow_validation import (
    validate_local_train_config,
    validate_production_train_config,
)
from lite_llm.local_smoke import TOKENS_PER_SHARD, VOCAB_SIZE, build_tokens
from lite_llm.modeling import (
    LiteLlmForCausalLM,
    RotaryEmbedding,
    _apply_rotary_pos_emb,
)
from lite_llm.token_storage import (
    count_tokens_in_file,
    existing_token_files,
    flush_token_shard,
    next_shard_index,
)
from lite_llm.train_runner import find_last_checkpoint

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_prepare_data_module():
    path = os.path.join(PROJECT_ROOT, "scripts", "production", "prepare_data.py")
    spec = importlib.util.spec_from_file_location("production_prepare_data", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_tiny_config(**overrides) -> LiteLlmConfig:
    defaults = dict(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        attention_type="mla",
        num_attention_heads=4,
        head_dim=16,
        q_lora_rank=16,
        kv_lora_rank=16,
        max_position_embeddings=64,
    )
    defaults.update(overrides)
    return LiteLlmConfig(**defaults)


class TrainingFixesTest(unittest.TestCase):
    def test_gradient_checkpointing_enable_supports_backward(self):
        model = LiteLlmForCausalLM(_make_tiny_config())
        model.gradient_checkpointing_enable()
        model.train()

        # Verify propagation: flag and func must both reach the inner model.
        self.assertTrue(model.model.gradient_checkpointing)
        self.assertTrue(hasattr(model.model, "_gradient_checkpointing_func"))

        input_ids = torch.randint(0, model.config.vocab_size, (2, 16))
        loss = model(input_ids, labels=input_ids).loss
        loss.backward()

        self.assertTrue(model.is_gradient_checkpointing)
        self.assertIsNotNone(model.model.embed_tokens.weight.grad)
        self.assertIsNotNone(model.model.layers[0].attention.q_up_proj.weight.grad)

    def test_weight_tying_shares_storage(self):
        model = LiteLlmForCausalLM(_make_tiny_config())
        self.assertIs(
            model.lm_head.weight,
            model.model.embed_tokens.weight,
        )

    def test_residual_projections_are_downscaled_at_init(self):
        torch.manual_seed(42)
        config = _make_tiny_config()
        model = LiteLlmForCausalLM(config)
        # With initializer_range=0.02 and the 1/sqrt(2L) residual scaling,
        # o_proj / down_proj should have noticeably smaller std than q_up_proj.
        q_std = model.model.layers[0].attention.q_up_proj.weight.std().item()
        o_std = model.model.layers[0].attention.o_proj.weight.std().item()
        down_std = model.model.layers[0].ffn.down_proj.weight.std().item()
        self.assertLess(o_std, q_std * 0.6)
        self.assertLess(down_std, q_std * 0.6)

    def test_training_loss_decreases_on_overfit_batch(self):
        torch.manual_seed(0)
        model = LiteLlmForCausalLM(_make_tiny_config())
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        ids = torch.randint(0, model.config.vocab_size, (2, 16))

        model.train()
        losses = []
        for _ in range(12):
            opt.zero_grad()
            out = model(ids, labels=ids)
            out.loss.backward()
            opt.step()
            losses.append(out.loss.item())

        self.assertLess(losses[-1], losses[0] - 1.0)

    def test_training_loss_decreases_without_qk_norm(self):
        torch.manual_seed(0)
        model = LiteLlmForCausalLM(_make_tiny_config(qk_norm=False))
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        ids = torch.randint(0, model.config.vocab_size, (2, 16))

        model.train()
        losses = []
        for _ in range(12):
            opt.zero_grad()
            out = model(ids, labels=ids)
            out.loss.backward()
            opt.step()
            losses.append(out.loss.item())

        self.assertLess(losses[-1], losses[0] - 1.0)

    def test_model_parameter_count(self):
        model = LiteLlmForCausalLM(_make_tiny_config())
        total = sum(p.numel() for p in model.parameters())
        # Weight tying: lm_head shares embed_tokens, so tied params count once.
        self.assertGreater(total, 0)
        self.assertLess(total, 500_000, "Tiny model should be well under 500K params")

    def test_generate_with_kv_cache_runs(self):
        model = LiteLlmForCausalLM(_make_tiny_config()).eval()
        ids = torch.randint(0, model.config.vocab_size, (2, 4))
        with torch.no_grad():
            y = model.generate(ids, max_new_tokens=6, do_sample=False)
        self.assertEqual(y.shape, (2, 10))

    def test_rotary_cos_sin_match_manual_formula(self):
        dim = 8
        max_seq_len = 16
        base = 10000.0
        rope = RotaryEmbedding(dim=dim, max_seq_len=max_seq_len, base=base)

        positions = torch.arange(max_seq_len)
        # Reference implementation: standard RoPE half-split angles.
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        freqs = torch.outer(positions.float(), inv_freq)
        expected_cos = torch.cat((freqs, freqs), dim=-1).cos()
        expected_sin = torch.cat((freqs, freqs), dim=-1).sin()

        dummy = torch.zeros(1, 1, max_seq_len, dim)
        cos, sin = rope(dummy, positions)
        self.assertTrue(torch.allclose(cos, expected_cos, atol=1e-6))
        self.assertTrue(torch.allclose(sin, expected_sin, atol=1e-6))

    def test_apply_rotary_preserves_vector_norm(self):
        torch.manual_seed(1)
        q = torch.randn(1, 2, 4, 8)
        k = torch.randn(1, 2, 4, 8)
        rope = RotaryEmbedding(dim=8, max_seq_len=8, base=10000.0)
        cos, sin = rope(q, torch.arange(4))
        q2, k2 = _apply_rotary_pos_emb(q, k, cos, sin)
        self.assertTrue(torch.allclose(q.norm(dim=-1), q2.norm(dim=-1), atol=1e-5))
        self.assertTrue(torch.allclose(k.norm(dim=-1), k2.norm(dim=-1), atol=1e-5))

    def test_find_last_checkpoint_uses_numeric_step_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "checkpoint-9"))
            os.makedirs(os.path.join(tmpdir, "checkpoint-200"))
            os.makedirs(os.path.join(tmpdir, "checkpoint-1000"))
            os.makedirs(os.path.join(tmpdir, "not-a-checkpoint"))

            last_checkpoint = find_last_checkpoint(tmpdir)

            self.assertEqual(
                last_checkpoint,
                os.path.join(tmpdir, "checkpoint-1000"),
            )

    def test_data_collator_creates_labels_from_input_ids(self):
        collator = DataCollatorForPretraining()
        features = [
            {"input_ids": torch.tensor([1, 2, 3])},
            {"input_ids": torch.tensor([4, 5, 6])},
        ]
        batch = collator(features)

        self.assertEqual(batch["input_ids"].shape, (2, 3))
        self.assertEqual(batch["labels"].shape, (2, 3))
        # Labels must be equal to input_ids for next-token prediction.
        self.assertTrue(torch.equal(batch["input_ids"], batch["labels"]))

    def test_load_tokenized_dataset_packs_across_file_boundaries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            np.save(os.path.join(tmpdir, "part_a.npy"), np.arange(0, 6, dtype=np.int32))
            np.save(os.path.join(tmpdir, "part_b.npy"), np.arange(6, 14, dtype=np.int32))

            dataset = load_tokenized_dataset(tmpdir, max_seq_length=4)

            self.assertEqual(len(dataset), 3)
            self.assertTrue(torch.equal(dataset[0]["input_ids"], torch.tensor([0, 1, 2, 3])))
            self.assertTrue(torch.equal(dataset[1]["input_ids"], torch.tensor([4, 5, 6, 7])))
            self.assertTrue(torch.equal(dataset[2]["input_ids"], torch.tensor([8, 9, 10, 11])))

    def test_split_train_val_uses_tail_as_eval(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            np.save(os.path.join(tmpdir, "s.npy"), np.arange(0, 100, dtype=np.int32))

            train_ds, val_ds = split_train_val(
                tmpdir, max_seq_length=10, val_fraction=0.2, max_val_tokens=None
            )
            self.assertEqual(len(train_ds), 8)
            self.assertEqual(len(val_ds), 2)
            # Val starts at token 80 (tail slice).
            self.assertTrue(
                torch.equal(val_ds[0]["input_ids"], torch.arange(80, 90, dtype=torch.long))
            )

    def test_split_train_val_returns_none_when_no_eval_tokens(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            np.save(os.path.join(tmpdir, "s.npy"), np.arange(0, 20, dtype=np.int32))
            train_ds, val_ds = split_train_val(tmpdir, max_seq_length=10, val_fraction=0.0)
            self.assertEqual(len(train_ds), 2)
            self.assertIsNone(val_ds)

    def test_tokenize_and_save_inserts_eos_between_documents(self):
        fake_tok = MagicMock()
        fake_tok.eos_token_id = 7
        fake_tok.encode.side_effect = lambda text, add_special_tokens: {
            "doc one": [1, 2, 3],
            "doc two": [4, 5],
        }[text]

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "out.npy")
            tokens = tokenize_and_save(["doc one", "doc two"], fake_tok, out_path)

        self.assertTrue((tokens == np.array([1, 2, 3, 7, 4, 5, 7], dtype=np.int32)).all())

    def test_prepare_data_flushes_into_append_only_shards(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            shard_idx = flush_token_shard([1, 2, 3], "math", tmpdir, 0)
            shard_idx = flush_token_shard([4, 5], "math", tmpdir, shard_idx)

            files = existing_token_files("math", tmpdir)

            self.assertEqual(shard_idx, 2)
            self.assertEqual(next_shard_index("math", tmpdir), 2)
            self.assertEqual(
                [os.path.basename(path) for path in files],
                ["math-00000.npy", "math-00001.npy"],
            )
            self.assertEqual([count_tokens_in_file(path) for path in files], [3, 2])

    def test_production_prepare_orders_files_with_stable_shuffle(self):
        prep = _load_prepare_data_module()
        files = [{"path": f"data/{i:03d}.parquet", "size": i} for i in range(10)]
        spec = {
            "name": "sample",
            "hf_path": "example/repo",
            "sampling": {"seed": 123, "file_order": "shuffled"},
        }

        first = prep._order_data_files(files, spec)
        second = prep._order_data_files(list(reversed(files)), spec)

        self.assertEqual(first, second)
        self.assertNotEqual([f["path"] for f in first], sorted(f["path"] for f in files))

    def test_production_prepare_inline_filters_use_text_and_quality(self):
        prep = _load_prepare_data_module()
        text = "中文内容" * 80
        cfg = {
            "min_chinese_ratio": 0.30,
            "min_length": 100,
            "max_length": 1000,
            "min_quality_score": 2,
            "quality_columns": ["score"],
        }

        self.assertTrue(prep._passes_filter_config({"score": 2.5}, text, cfg))
        self.assertFalse(prep._passes_filter_config({"score": 1.0}, text, cfg))
        self.assertFalse(prep._passes_filter_config({"score": 3.0}, "mostly ascii", cfg))

    def test_production_prepare_extracts_first_available_text_column(self):
        prep = _load_prepare_data_module()
        spec = {"text_columns": ["text", "content"]}

        self.assertEqual(
            prep._extract_text({"content": "fallback text"}, spec),
            "fallback text",
        )

    def test_production_prepare_normalizes_recipe_aliases(self):
        prep = _load_prepare_data_module()

        specs = prep.build_specs({
            "sampling": {"seed": 1},
            "datasets": [
                {
                    "name": "sample",
                    "source": "org/repo",
                    "subset": "subset-name",
                    "text_field": "body",
                    "target_tokens": 100,
                    "filters": {"min_chars": 10, "max_chars": 100, "add_eos": True},
                },
                {
                    "name": "disabled",
                    "source": None,
                    "target_tokens": 100,
                    "enabled": False,
                },
            ],
        })

        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0]["hf_path"], "org/repo")
        self.assertEqual(specs[0]["config"], "subset-name")
        self.assertEqual(specs[0]["text_column"], "body")
        self.assertEqual(specs[0]["filter_config"]["min_length"], 10)
        self.assertEqual(specs[0]["filter_config"]["max_length"], 100)
        self.assertTrue(specs[0]["add_eos"])

    def test_local_smoke_tokens_stay_in_vocab(self):
        tokens = build_tokens()

        self.assertEqual(len(tokens), TOKENS_PER_SHARD)
        self.assertGreaterEqual(int(tokens.min()), 0)
        self.assertLess(int(tokens.max()), VOCAB_SIZE)

    def test_local_config_isolated_from_production_paths(self):
        with open(os.path.join(PROJECT_ROOT, "configs", "local", "train.yaml"), "r") as f:
            train_cfg = yaml.safe_load(f)
        with open(os.path.join(PROJECT_ROOT, "configs", "local", "model.yaml"), "r") as f:
            model_cfg = yaml.safe_load(f)

        validate_local_train_config(train_cfg, model_cfg)

    def test_production_config_isolated_from_local_paths(self):
        with open(os.path.join(PROJECT_ROOT, "configs", "production", "train.yaml"), "r") as f:
            train_cfg = yaml.safe_load(f)
        with open(os.path.join(PROJECT_ROOT, "configs", "production", "model.yaml"), "r") as f:
            model_cfg = yaml.safe_load(f)

        validate_production_train_config(train_cfg, model_cfg)

    def test_zh_first_production_train_configs_are_isolated(self):
        with open(os.path.join(PROJECT_ROOT, "configs", "production", "model.yaml"), "r") as f:
            model_cfg = yaml.safe_load(f)

        for filename in ("train_zh_first_3b.yaml", "train_zh_first_20b.yaml"):
            with open(os.path.join(PROJECT_ROOT, "configs", "production", filename), "r") as f:
                train_cfg = yaml.safe_load(f)
            validate_production_train_config(train_cfg, model_cfg)


if __name__ == "__main__":
    unittest.main()
