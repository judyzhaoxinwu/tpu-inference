# Copyright 2025 Google LLC


"""Wrapper for AXLearn models."""

import jax
import jax.numpy as jnp

from flax import nnx
from jax.sharding import Mesh
from transformers import LlamaConfig
from vllm.config import VllmConfig
from axlearn.common.module import functional as F
from axlearn.experiments.text.gpt.fuji import model_config as fuji_model_config


class AxLearnForCausalLM(nnx.Module):
  def __init__(self, vllm_config: VllmConfig, rng_key: jax.Array, mesh: Mesh) -> None:
    model_config_hf = vllm_config.model_config.hf_config

    # Create a fuji model config
    self.fuji_config = fuji_model_config(
        num_layers=model_config_hf.num_hidden_layers,
        hidden_dim=model_config_hf.hidden_size,
        num_kv_heads=model_config_hf.num_key_value_heads,
        vocab_size=model_config_hf.vocab_size,
        rope_theta=model_config_hf.rope_theta,
        shared_lm_head=model_config_hf.tie_word_embeddings,
        dropout_rate=0.0,
        ffn_dim=model_config_hf.intermediate_size,
        flash_attention=model_config_hf.use_flash_attention_2,
        stack_cfg=None,
        pad_token_id=model_config_hf.pad_token_id,
        eos_token_id=model_config_hf.eos_token_id,
    )

    self.model = self.fuji_config.instantiate(
        parent=None, name="fuji", rng_key=rng_key, mesh=mesh, dtype=jnp.float16
    )

  def __call__(
    self, 
    kv_caches: List[jax.Array],
    input_ids: jax.Array,
    attention_metadata: AttentionMetadata,
    _input_embeds=None, 
    _input_positions=None,
    _layer_name_to_kv_cache=None,
    _lora_metadata=None,
    intermediate_tensors: JaxIntermediateTensors | None = None,
    _is_first_rank: bool | None = None, 
    _is_last_rank: bool | None = None,
    *args,
  ) -> Tuple[List[jax.Array], jax.Array | JaxIntermediateTensors, 
             List[jax.Array]. Optional[jax.Array]]:
             pass

  def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
    pass

  def load_weights(self, rng_key: jax.Array):
    pass
