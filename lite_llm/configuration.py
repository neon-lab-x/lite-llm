from transformers import PretrainedConfig


class LiteLlmConfig(PretrainedConfig):
    """Config for a decoder-only causal LM with MLA attention.

    Notes
    -----
    - ``attention_type`` is currently fixed to ``"mla"``.
    - ``q_lora_rank`` and ``kv_lora_rank`` control the latent bottleneck ranks
      used by the query and KV projections.
    - ``qk_norm`` toggles per-head RMSNorm on Q/K before applying RoPE.
    - ``tie_word_embeddings`` ties the lm_head weight with the input embedding.
    - ``initializer_range`` controls weight init std (Llama-style N(0, std)).
    """

    model_type = "lite_llm"

    def __init__(
        self,
        vocab_size=248320,
        hidden_size=2048,
        intermediate_size=5504,
        num_hidden_layers=24,
        num_attention_heads=16,
        head_dim=128,
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_type="mla",
        q_lora_rank=512,
        kv_lora_rank=192,
        qk_norm=True,
        initializer_range=0.02,
        tie_word_embeddings=True,
        use_cache=True,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_type = attention_type
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_norm = qk_norm
        self.initializer_range = initializer_range
        self.use_cache = use_cache

        if attention_type != "mla":
            raise ValueError(
                "Only MLA attention is currently supported: "
                f"got attention_type={attention_type!r}."
            )
        if hidden_size != num_attention_heads * head_dim:
            raise ValueError(
                "hidden_size must equal num_attention_heads * head_dim: "
                f"got hidden_size={hidden_size}, heads={num_attention_heads}, "
                f"head_dim={head_dim}."
            )
        if q_lora_rank <= 0 or q_lora_rank > hidden_size:
            raise ValueError(
                "q_lora_rank must be in the range (0, hidden_size]: "
                f"got q_lora_rank={q_lora_rank}, hidden_size={hidden_size}."
            )
        if kv_lora_rank <= 0 or kv_lora_rank > hidden_size:
            raise ValueError(
                "kv_lora_rank must be in the range (0, hidden_size]: "
                f"got kv_lora_rank={kv_lora_rank}, hidden_size={hidden_size}."
            )
