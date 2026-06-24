# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Wrapper for AXLearn models."""

import threading
from typing import Any, List, Optional, Tuple

import jax
import jax.numpy as jnp
# AxLearn imports moved locally inside serving methods to ensure mesh context binding
from flax import nnx
from jax.sharding import Mesh
from vllm.config import VllmConfig

from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.logger import init_logger
from tpu_inference.models.common.model_loader import register_model
from tpu_inference.models.jax.jax_intermediate_tensor import \
    JaxIntermediateTensors

logger = init_logger(__name__)

# Thread-local storage to inject active vLLM Key-Value cache arrays and attention metadata
# directly into AxLearn's internal JAX SelfAttention evaluation layer.
_vllm_context = threading.local()
_vllm_context.kv_caches = None
_vllm_context.attention_metadata = None
_vllm_context.layer_index = 0

# Global flag to dynamically enable/disable the HuggingFace split-half RoPE monkey-patch
# based on whether the loaded model's GCS checkpoint has its weights pre-permuted or not.
_USE_SPLIT_HALF_ROPE = False
_orig_apply_rotary_position_embeddings = None


def register():
    logger.info(
        "Registering AxLearnForCausalLM model with tpu_inference and vllm.")
    register_model("AxLearnForCausalLM", AxLearnForCausalLM)
    logger.info("Successfully registered AxLearnForCausalLM model.")


def _sanitize_partition_spec(spec, allowed_axes):
    if spec is None:
        return None
    from jax.sharding import PartitionSpec
    if isinstance(spec, PartitionSpec):
        new_axes = []
        for axis in spec:
            if isinstance(axis, tuple):
                # Filter out axes that are not in the serving mesh
                filtered = tuple(a for a in axis if a in allowed_axes)
                if len(filtered) == 1:
                    new_axes.append(filtered[0])
                elif len(filtered) > 1:
                    new_axes.append(filtered)
                else:
                    new_axes.append(None)
            else:
                new_axes.append(axis if axis in allowed_axes else None)
        return PartitionSpec(*new_axes)
    return spec


def _recursive_sanitize_specs(cfg, allowed_axes, mha_cls=None, gqa_cls=None):
    from axlearn.common.attention import (GroupedQueryAttention,
                                          TransformerAttentionLayer)
    from axlearn.common.config import ConfigBase
    from axlearn.common.repeat import Repeat
    if isinstance(cfg, ConfigBase):
        if isinstance(cfg, Repeat.Config):
            cfg.unroll = True
        if mha_cls and gqa_cls and isinstance(
                cfg, TransformerAttentionLayer.Config):
            is_gqa = issubclass(cfg.attention.klass, GroupedQueryAttention)
            target_cls = gqa_cls if is_gqa else mha_cls
            cfg.attention.set(klass=target_cls)
        for key in cfg.keys():
            val = getattr(cfg, key)
            if isinstance(val, dict):
                from jax.sharding import PartitionSpec
                new_dict = {}
                for k, v in val.items():
                    if isinstance(v, PartitionSpec):
                        new_dict[k] = _sanitize_partition_spec(v, allowed_axes)
                    else:
                        new_dict[k] = v
                        _recursive_sanitize_specs(v, allowed_axes, mha_cls,
                                                  gqa_cls)
                cfg.set(**{key: new_dict})
            elif isinstance(val, (list, tuple)):
                from jax.sharding import PartitionSpec
                new_list = []
                for item in val:
                    if isinstance(item, PartitionSpec):
                        new_list.append(
                            _sanitize_partition_spec(item, allowed_axes))
                    else:
                        new_list.append(item)
                        _recursive_sanitize_specs(item, allowed_axes, mha_cls,
                                                  gqa_cls)
                cfg.set(**{key: type(val)(new_list)})
            elif hasattr(val, 'klass') or isinstance(val, ConfigBase):
                _recursive_sanitize_specs(val, allowed_axes, mha_cls, gqa_cls)
            else:
                from jax.sharding import PartitionSpec
                if isinstance(val, PartitionSpec):
                    cfg.set(
                        **{key: _sanitize_partition_spec(val, allowed_axes)})


def _recursive_set_block_size(cfg):
    from axlearn.common.config import ConfigBase
    from axlearn.common.flash_attention.layer import FlashAttention
    if isinstance(cfg, ConfigBase):
        if isinstance(cfg, FlashAttention.Config):
            logger.info(
                "Setting tpu_block_size=128 to fit within TPU v6e VMEM limit")
            cfg.tpu_block_size = 128
            if getattr(cfg, 'backend_overrides', None) is not None:
                cfg.backend_overrides["splash_block_q"] = 128
                cfg.backend_overrides["splash_block_kv"] = 128
                cfg.backend_overrides["splash_block_kv_compute"] = 128
        for key in cfg.keys():
            val = getattr(cfg, key)
            if isinstance(val, dict):
                for v in val.values():
                    _recursive_set_block_size(v)
            elif isinstance(val, (list, tuple)):
                for item in val:
                    _recursive_set_block_size(item)
            elif hasattr(val, 'klass') or isinstance(val, ConfigBase):
                _recursive_set_block_size(val)


def _recursive_set_outer_batch(cfg, outer_batch_size):
    from axlearn.common.config import ConfigBase

    if isinstance(cfg, ConfigBase):
        if hasattr(cfg, "outer_batch"):
            logger.info(
                f"Dynamically setting MoE outer_batch to {outer_batch_size} to match serving mesh"
            )
            cfg.outer_batch = outer_batch_size
        for key in cfg.keys():
            val = getattr(cfg, key)
            if isinstance(val, dict):
                for v in val.values():
                    _recursive_set_outer_batch(v, outer_batch_size)
            elif isinstance(val, (list, tuple)):
                for item in val:
                    _recursive_set_outer_batch(item, outer_batch_size)
            elif hasattr(val, "klass") or isinstance(val, ConfigBase):
                _recursive_set_outer_batch(val, outer_batch_size)


class AxLearnForCausalLM(nnx.Module):

    def __init__(self, vllm_config: VllmConfig, rng_key: jax.Array,
                 mesh: Mesh) -> None:
        self.vllm_config = vllm_config
        self._qk_norm_remap_mode = None

        # Re-map vLLM's physical mesh axis names to AxLearn's logical axis names
        vllm_axis_to_axlearn = {
            "data": "data",
            "model": "model",
            "expert": "expert",
            "attn_dp_expert": "fsdp",
            "attn_dp": "seq",
            "dcp": "pipeline"
        }
        axlearn_axis_names = tuple(
            vllm_axis_to_axlearn.get(name, name) for name in mesh.axis_names)
        self.mesh = jax.sharding.Mesh(mesh.devices, axlearn_axis_names)

        # Register the remapped mesh globally in AxLearn's physical mesh fallback
        from axlearn.common.utils import thread_resources
        thread_resources.env = thread_resources.env._replace(
            physical_mesh=self.mesh)

        # Dynamic imports inside active JAX Mesh context manager to force shard_map axis binding!
        with self.mesh:
            # Monkey-patch AxLearn's native interleaved RoPE with HuggingFace-compatible split-half RoPE.
            # This is required if the converted checkpoint did not have its Q/K weight matrices permuted.
            import axlearn.common.attention as axlearn_attention
            from axlearn.common.attention import (
                ForwardMode, FusedGroupedQKVLinear, FusedQKVLinear,
                GroupedQueryAttention, MultiheadAttention, RoFormerQKVLinear)
            from axlearn.common.layers import RMSNorm
            from axlearn.common.utils import Tensor
            global _orig_apply_rotary_position_embeddings
            if _orig_apply_rotary_position_embeddings is None:
                _orig_apply_rotary_position_embeddings = axlearn_attention.apply_rotary_position_embeddings

            def split_half_apply_rotary_position_embeddings(
                *,
                query: jax.Array,
                key: jax.Array,
                value: jax.Array,
                sinusoidal_pos: jax.Array,
                rotary_key: bool,
                rotary_value: bool,
            ) -> Tuple[jax.Array, jax.Array, jax.Array]:
                global _USE_SPLIT_HALF_ROPE
                if not _USE_SPLIT_HALF_ROPE:
                    # Fall back directly to AxLearn's native interleaved RoPE!
                    return _orig_apply_rotary_position_embeddings(
                        query=query,
                        key=key,
                        value=value,
                        sinusoidal_pos=sinusoidal_pos,
                        rotary_key=rotary_key,
                        rotary_value=rotary_value,
                    )

                sin, cos = jnp.split(sinusoidal_pos, 2, axis=-1)
                cos_pos = jnp.concatenate([cos, cos], axis=-1)
                sin_pos = jnp.concatenate([sin, sin], axis=-1)

                def rotate_half(x):
                    half_dim = x.shape[-1] // 2
                    return jnp.concatenate(
                        [-x[..., half_dim:], x[..., :half_dim]], axis=-1)

                query = query * cos_pos + rotate_half(query) * sin_pos
                if rotary_key:
                    key = key * cos_pos + rotate_half(key) * sin_pos
                if rotary_value:
                    value = value * cos_pos + rotate_half(value) * sin_pos
                return query, key, value

            axlearn_attention.apply_rotary_position_embeddings = split_half_apply_rotary_position_embeddings
            logger.info(
                "=== [MONKEY PATCH] === Successfully registered dynamic HuggingFace/AxLearn RoPE router!"
            )
            from axlearn.experiments.text.gpt.c4_trainer import \
                named_trainer_configs as c4_configs
            from axlearn.experiments.text.gpt.common import \
                model_config as common_model_config
            from axlearn.experiments.text.gpt.pajama_trainer import \
                named_trainer_configs as pajama_configs

            class VllmAttentionMixin:

                def _forward_for_mode(
                    self,
                    *,
                    mode: ForwardMode,
                    query: Tensor,
                    key=None,
                    value=None,
                    kv_state=None,
                    attention_logit_biases=None,
                    segment_ids=None,
                    query_positions=None,
                    cached_states=None,
                    return_aux=None,
                    page_pool=None,
                ):
                    query_positions = jnp.arange(
                        query.shape[1]
                    )[None] if query_positions is None else query_positions
                    q_proj, k_proj, v_proj = self.i_proj(
                        query, query_positions=query_positions)

                    kv_cache_array = _vllm_context.kv_caches[
                        _vllm_context.layer_index]
                    md = _vllm_context.attention_metadata

                    import jax

                    from tpu_inference.layers.common.attention_interface import \
                        attention
                    mesh = jax.sharding.get_abstract_mesh()
                    q_proj_3d = jnp.squeeze(q_proj, axis=1)
                    k_proj_3d = jnp.squeeze(k_proj, axis=1)
                    v_proj_3d = jnp.squeeze(v_proj, axis=1)

                    # Cast inputs to match kv_cache dtype (bfloat16) to prevent kernel mismatch
                    cache_dtype = kv_cache_array.dtype
                    q_proj_3d = q_proj_3d.astype(cache_dtype)
                    k_proj_3d = k_proj_3d.astype(cache_dtype)
                    v_proj_3d = v_proj_3d.astype(cache_dtype)

                    new_kv_cache, outputs_3d = attention(
                        kv_cache_array,
                        q_proj_3d,
                        k_proj_3d,
                        v_proj_3d,
                        md,
                        mesh,
                        self.per_head_dim(),
                    )
                    outputs = jnp.expand_dims(outputs_3d,
                                              axis=1).astype(q_proj.dtype)

                    _vllm_context.kv_caches[
                        _vllm_context.layer_index] = new_kv_cache
                    _vllm_context.layer_index += 1

                    out = self.o_proj(outputs)
                    return dict(), self.Output(data=out)

            class VllmMultiheadAttention(VllmAttentionMixin,
                                         MultiheadAttention):
                pass

            class VllmGroupedQueryAttention(VllmAttentionMixin,
                                            GroupedQueryAttention):
                pass

        model_config_hf = vllm_config.model_config.hf_config
        if (hasattr(model_config_hf, "thinker_config")
                and hasattr(model_config_hf.thinker_config, "text_config")):
            model_config_hf = model_config_hf.thinker_config.text_config
        elif hasattr(model_config_hf, "text_config"):
            model_config_hf = model_config_hf.text_config

        logger.info(
            f"=== [HF CONFIG DEBUG] ===\n{model_config_hf}\n========================="
        )

        global _USE_SPLIT_HALF_ROPE
        # Since both the 0.6B and 30B checkpoints on GCS are now successfully converted
        # using the updated offline converter script (which permutes Q/K weights to interleaved format),
        # we set _USE_SPLIT_HALF_ROPE = False for all models. JAX will run native interleaved RoPE.
        _USE_SPLIT_HALF_ROPE = False
        logger.info(
            "=== [ROPE SWITCH] === All active GCS checkpoints are offline-permuted. Running 100% native AxLearn interleaved RoPE."
        )

        self.hidden_dim = getattr(model_config_hf, "hidden_size",
                                  getattr(model_config_hf, "hidden_dim", None))
        axlearn_cfg = getattr(vllm_config, "additional_config",
                              {}).get("axlearn_config", {})
        hf_vocab = getattr(model_config_hf, "vocab_size", None)
        if hf_vocab == 152064:
            hf_vocab = 151936
        self.vocab_size = axlearn_cfg.get("vocab_size", hf_vocab)
        model_name = axlearn_cfg.get("model_name", None)

        configs_map = {}
        configs_map.update(c4_configs())
        configs_map.update(pajama_configs())
        use_registry = False
        if model_name and model_name in configs_map:
            # We bypass the AxLearn registry configs map for Qwen models
            # to force model-agnostic mapping from the Hugging Face config.
            # This is essential to prevent shape/parameter tree mismatches with converted HF checkpoints.
            if "qwen" in model_name.lower():
                logger.info(
                    f"Bypassing AxLearn registry for Qwen model '{model_name}' to map model-agnostically from HF config."
                )
            else:
                use_registry = True

        if use_registry:
            logger.info(
                f"Instantiating model structure directly from AxLearn registry: {model_name}"
            )
            trainer_cfg = configs_map[model_name]()
            self.axlearn_model_config = trainer_cfg.model.set(name=model_name)
        else:
            logger.info(
                f"Named config '{model_name}' not found in AxLearn registry. Mapping properties model-agnostically from HF config."
            )
            # 1. Resolve Grouped Query Attention (GQA) and attention hidden dim parameters
            num_kv_heads = getattr(model_config_hf, "num_key_value_heads",
                                   None)
            per_head_dim = getattr(model_config_hf, "head_dim", None)
            if per_head_dim is None:
                per_head_dim = getattr(model_config_hf, "per_head_dim", 128)
            atten_hidden_dim = model_config_hf.num_attention_heads * per_head_dim

            if num_kv_heads and num_kv_heads != model_config_hf.num_attention_heads:
                atten_cfg = VllmGroupedQueryAttention.default_config().set(
                    hidden_dim=atten_hidden_dim)
                atten_input_linear = FusedGroupedQKVLinear.default_config(
                ).set(num_kv_heads=num_kv_heads)
            else:
                atten_cfg = VllmMultiheadAttention.default_config().set(
                    hidden_dim=atten_hidden_dim)
                atten_input_linear = FusedQKVLinear.default_config()

            from axlearn.common.attention import (ScaleKey, ScaleQuery,
                                                  constant_scale_fn)
            from axlearn.common.config import config_for_function
            norm_cfg = RMSNorm.default_config().set(
                eps=getattr(model_config_hf, "rms_norm_eps", 1e-6),
                forward_dtype=jnp.float32,
            )

            # 2. Setup Rotary Position Embeddings (RoPE) and Q/K Linear
            # Configure QK-Norm scales on the inner projection layer (RoFormerQKVLinear)
            # so they execute BEFORE RoPE, matching the mathematically correct order of operations.
            attention_qkv_linear = RoFormerQKVLinear.default_config().set(
                input_linear=atten_input_linear,
                rotary_value=False,
                query_scale=ScaleQuery.default_config().set(
                    norm=norm_cfg.clone(),
                    scale_factor=config_for_function(constant_scale_fn).set(
                        value=1.0)),
                key_scale=ScaleKey.default_config().set(norm=norm_cfg.clone()),
            )
            # Robustly extract rope_theta, checking rope_scaling and rope_parameters dicts
            rope_theta = getattr(model_config_hf, "rope_theta", None)
            if rope_theta is None:
                rope_scaling = getattr(model_config_hf, "rope_scaling", None)
                if isinstance(rope_scaling, dict):
                    rope_theta = rope_scaling.get("rope_theta", None)
            if rope_theta is None:
                rope_params = getattr(model_config_hf, "rope_parameters", None)
                if isinstance(rope_params, dict):
                    rope_theta = rope_params.get("rope_theta", None)
            if rope_theta is None:
                rope_theta = 10000.0

            attention_qkv_linear.rope_pos_emb_layer.set(
                theta=float(rope_theta))

            # 3. Setup MoE parameters dynamically (if expert keys are present)
            num_experts = getattr(
                model_config_hf, "num_local_experts",
                getattr(model_config_hf, "num_experts", None))

            # Both dense (0.6B) and MoE (30B) Qwen checkpoints on GCS are now successfully aligned
            # to the inner layout (i_proj/scale_query), matching the JAX serving structure natively.
            # No serving-time remapping is needed for either model!
            self._qk_norm_remap_mode = None
            ffn_layer_types = None
            expert_cfg = None
            if num_experts is not None:
                from axlearn.common.mixture_of_experts import (
                    TopKGating, TransformerFeedForwardMoE)
                ffn_layer_types = ["sparse"]
                num_experts_per_token = getattr(
                    model_config_hf, "num_experts_per_tok",
                    getattr(model_config_hf, "num_experts_per_token", 8))
                from axlearn.common.utils import PartitionSpec
                expert_cfg = TransformerFeedForwardMoE.default_config().set(
                    num_experts=num_experts,
                    num_groups=1,
                    dim_to_mesh_axis_map={
                        "me":
                        PartitionSpec(None, None),
                        "emh":
                        PartitionSpec("model", None, None),
                        "ehm":
                        PartitionSpec("model", None, None),
                        "ogsm":
                        PartitionSpec("data", "expert", None, "model"),
                        "ogsec":
                        PartitionSpec("data", "expert", None, None, None),
                        "oegcm":
                        PartitionSpec("data", "expert", None, None, "model"),
                        "ogecm":
                        PartitionSpec("data", "expert", None, None, "model"),
                        "oegch":
                        PartitionSpec("data", "expert", None, None, "model"),
                    },
                    gating=TopKGating.default_config().set(
                        num_experts_per_token=num_experts_per_token,
                        train_capacity_factor=0,
                        eval_capacity_factor=float(num_experts),
                    ),
                )

            from axlearn.common import decoder

            # Resolve the expert FFN dimension.
            # We use moe_intermediate_size (if present) for MoE models (e.g., 768),
            # and intermediate_size (e.g., 6144) for dense models.
            model_type = getattr(model_config_hf, "model_type", "")
            ffn_dim = getattr(
                model_config_hf, "moe_intermediate_size",
                getattr(model_config_hf, "intermediate_size", None))
            logger.info(
                f"=== [FFN DIM DEBUG] === ffn_dim: {ffn_dim} | model_type: {model_type}"
            )
            self.axlearn_model_config = common_model_config(
                num_layers=model_config_hf.num_hidden_layers,
                hidden_dim=model_config_hf.hidden_size,
                num_heads=model_config_hf.num_attention_heads,
                vocab_size=self.vocab_size,
                activation_fn=("nn.silu", "linear"),  # SwiGLU
                ffn_dim=ffn_dim,
                normalization=RMSNorm.default_config().set(
                    eps=getattr(model_config_hf, "rms_norm_eps", 1e-6),
                    forward_dtype=jnp.float32,
                ),
                attention_cfg=atten_cfg,
                attention_qkv_linear=attention_qkv_linear,
                ffn_layer_types=ffn_layer_types,
                expert_cfg=expert_cfg,
                lm_head_cfg=None
                if getattr(model_config_hf, "tie_word_embeddings", False) else
                decoder.LmHead.default_config(),
                pad_token_id=model_config_hf.pad_token_id,
                eos_token_id=model_config_hf.eos_token_id,
            ).set(name=model_name or "axlearn_model")

            self.axlearn_model_config.decoder.output_norm = RMSNorm.default_config(
            ).set(
                input_dim=model_config_hf.hidden_size,
                eps=getattr(model_config_hf, "rms_norm_eps", 1e-6),
                forward_dtype=jnp.float32,
            )

        abstract_mesh = jax.sharding.get_abstract_mesh()
        allowed_axes = abstract_mesh.axis_names if not abstract_mesh.empty else (
            "model", "expert")
        logger.info(
            f"Sanitizing model PartitionSpecs to match active compilation mesh axes: {allowed_axes}"
        )
        _recursive_sanitize_specs(self.axlearn_model_config, allowed_axes,
                                  VllmMultiheadAttention,
                                  VllmGroupedQueryAttention)
        _recursive_set_block_size(self.axlearn_model_config)

        # Dynamically calculate MoE outer batch size from active serving mesh
        outer_batch_size = 1
        for axis in ("data", "fsdp"):
            if axis in self.mesh.shape:
                outer_batch_size *= self.mesh.shape[axis]
        _recursive_set_outer_batch(self.axlearn_model_config, outer_batch_size)

        # Explicitly enforce highly efficient layer-wise Remat boundaries to reduce XLA compilation temporaries from 18 GB down to 256 MB
        from axlearn.common.base_layer import RematSpec
        from axlearn.common.flash_attention.remat import \
            save_or_offload_flash_attention_policy
        try:
            self.axlearn_model_config.decoder.transformer.layer.set(
                remat_spec=RematSpec(
                    prevent_cse=False,
                    policy=save_or_offload_flash_attention_policy(),
                ))
        except AttributeError:
            pass

        logger.info(
            f"=== [MODEL CONFIG FFN DIM] === hidden_dim: {self.axlearn_model_config.decoder.transformer.layer.feed_forward.hidden_dim}"
        )
        with self.mesh:
            self.model = self.axlearn_model_config.instantiate(parent=None)

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
               List[jax.Array], Optional[jax.Array]]:

        import sys
        print("VLLM JAX CONTEXT MESH:",
              jax.sharding.get_abstract_mesh(),
              file=sys.stderr)
        print("VLLM MODEL MESH:", self.mesh, file=sys.stderr)

        _vllm_context.kv_caches = list(kv_caches)
        _vllm_context.attention_metadata = attention_metadata
        _vllm_context.layer_index = 0

        input_ids_2d = jnp.expand_dims(input_ids, axis=1)
        pos = attention_metadata.input_positions
        if pos.ndim > 1:
            pos = pos[0]
        positions_2d = jnp.expand_dims(pos, axis=1)

        input_batch = dict(
            input_ids=input_ids_2d,
            input_positions=positions_2d,
            input_segment_ids=jnp.ones_like(input_ids_2d),
        )

        inputs = dict(
            input_batch=input_batch,
            return_aux=True,
        )

        from axlearn.common.module import functional as F
        with self.mesh:
            outputs, output_collection = F(
                self.model,
                state=self.axlearn_state.value,
                method="forward",
                inputs=inputs,
                prng_key=jax.random.key(0),
                is_training=False,
            )

        loss, aux_outputs = outputs
        hidden_states = aux_outputs["hidden_states"]
        hidden_states = hidden_states.reshape((-1, hidden_states.shape[-1]))

        updated_kv_caches = list(_vllm_context.kv_caches)
        return updated_kv_caches, hidden_states, [], None

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        hidden_states_3d = jnp.expand_dims(hidden_states, axis=1)

        from axlearn.common.module import functional as F
        with self.mesh:
            if hasattr(self.model.decoder,
                       "lm_head") and self.model.decoder.lm_head is not None:
                logits, _ = F(
                    self.model.decoder.lm_head,
                    state=self.axlearn_state.value["decoder"]["lm_head"],
                    method="forward",
                    inputs=dict(x=hidden_states_3d),
                    prng_key=jax.random.key(0),  # Supply dummy key
                    is_training=False,
                )
            else:
                logits, _ = F(
                    self.model.decoder.emb,
                    state=self.axlearn_state.value["decoder"]["emb"],
                    method="attend",
                    inputs=dict(x=hidden_states_3d),
                    prng_key=jax.random.key(0),  # Supply dummy key
                    is_training=False,
                )
        return logits.squeeze(1)

    def load_weights(self, rng_key: jax.Array):
        ckpt_path = None
        if "axlearn_config" in self.vllm_config.additional_config:
            ckpt_path = self.vllm_config.additional_config[
                "axlearn_config"].get("ckpt_path", None)

        if ckpt_path is not None:
            logger.info(
                f"Streaming authentic checkpoint parameters from remote storage: {ckpt_path}"
            )

            from axlearn.common.checkpointer import CheckpointValidationType
            from axlearn.common.state_builder import (
                Builder, TensorStoreStateStorageBuilder)
            with self.mesh:
                target_specs = dict(
                    model=self.model.create_parameter_specs_recursively())

                # Remap target specs from inner to outer so the checkpointer
                # looks up the correct outer paths in the checkpoint directory.
                # Align the JAX model specs to match the GCS checkpoint layout
                if self._qk_norm_remap_mode == "outer_to_inner":

                    def remap_inner_to_outer(d):
                        if hasattr(d, "items"):
                            new_dict = {}
                            for k, v in d.items():
                                new_dict[k] = remap_inner_to_outer(v)
                            if "i_proj" in new_dict and hasattr(
                                    new_dict["i_proj"], "items"):
                                i_proj = new_dict["i_proj"]
                                scale_key = i_proj.pop("scale_key", None)
                                scale_query = i_proj.pop("scale_query", None)
                                if scale_key is not None:
                                    new_dict["scale_key"] = scale_key
                                if scale_query is not None:
                                    new_dict["scale_query"] = scale_query
                            return new_dict
                        if isinstance(d, list):
                            return [remap_inner_to_outer(x) for x in d]
                        if isinstance(d, tuple):
                            return tuple(remap_inner_to_outer(x) for x in d)
                        return d

                    target_specs = remap_inner_to_outer(target_specs)
                elif self._qk_norm_remap_mode == "inner_to_outer":

                    def remap_outer_to_inner(d):
                        if hasattr(d, "items"):
                            new_dict = {}
                            for k, v in d.items():
                                new_dict[k] = remap_outer_to_inner(v)
                            scale_key = new_dict.pop("scale_key", None)
                            scale_query = new_dict.pop("scale_query", None)
                            if scale_key is not None or scale_query is not None:
                                if "i_proj" not in new_dict:
                                    new_dict["i_proj"] = {}
                                if scale_key is not None:
                                    new_dict["i_proj"]["scale_key"] = scale_key
                                if scale_query is not None:
                                    new_dict["i_proj"][
                                        "scale_query"] = scale_query
                            return new_dict
                        if isinstance(d, list):
                            return [remap_outer_to_inner(x) for x in d]
                        if isinstance(d, tuple):
                            return tuple(remap_outer_to_inner(x) for x in d)
                        return d

                    target_specs = remap_outer_to_inner(target_specs)

                storage_builder = TensorStoreStateStorageBuilder.default_config(
                ).set(
                    name="storage",
                    dir=ckpt_path,
                    validation=CheckpointValidationType.
                    CONTAINS_STATE_UP_TO_DTYPE,
                ).instantiate(parent=None)

                built_state = storage_builder(
                    Builder.State(
                        step=0,
                        trainer_state=target_specs,
                        built_keys=set(),
                    ))
                init_state = built_state.trainer_state["model"]

                # Remap loaded parameters to match the active serving model layout
                if self._qk_norm_remap_mode == "outer_to_inner":

                    def to_mutable_dict_and_remap(d):
                        if hasattr(d, "items"):
                            new_dict = {}
                            for k, v in d.items():
                                new_dict[k] = to_mutable_dict_and_remap(v)
                            scale_key = new_dict.pop("scale_key", None)
                            scale_query = new_dict.pop("scale_query", None)
                            if scale_key is not None or scale_query is not None:
                                logger.info(
                                    f"=== [REMAP SUCCESS: OUTER -> INNER] === Remapped scales under: {list(new_dict.keys())}"
                                )
                                if "i_proj" not in new_dict:
                                    new_dict["i_proj"] = {}
                                if scale_key is not None:
                                    new_dict["i_proj"]["scale_key"] = scale_key
                                if scale_query is not None:
                                    new_dict["i_proj"][
                                        "scale_query"] = scale_query
                            return new_dict
                        if isinstance(d, list):
                            return [to_mutable_dict_and_remap(x) for x in d]
                        if isinstance(d, tuple):
                            return tuple(
                                to_mutable_dict_and_remap(x) for x in d)
                        return d

                    init_state = to_mutable_dict_and_remap(init_state)
                elif self._qk_norm_remap_mode == "inner_to_outer":

                    def to_mutable_dict_and_remap(d):
                        if hasattr(d, "items"):
                            new_dict = {}
                            for k, v in d.items():
                                new_dict[k] = to_mutable_dict_and_remap(v)
                            if "i_proj" in new_dict and hasattr(
                                    new_dict["i_proj"], "items"):
                                i_proj = new_dict["i_proj"]
                                scale_key = i_proj.pop("scale_key", None)
                                scale_query = i_proj.pop("scale_query", None)
                                if scale_key is not None or scale_query is not None:
                                    logger.info(
                                        f"=== [REMAP SUCCESS: INNER -> OUTER] === Remapped scales under: {list(new_dict.keys())}"
                                    )
                                    if scale_key is not None:
                                        new_dict["scale_key"] = scale_key
                                    if scale_query is not None:
                                        new_dict["scale_query"] = scale_query
                            return new_dict
                        if isinstance(d, list):
                            return [to_mutable_dict_and_remap(x) for x in d]
                        if isinstance(d, tuple):
                            return tuple(
                                to_mutable_dict_and_remap(x) for x in d)
                        return d

                    init_state = to_mutable_dict_and_remap(init_state)
        else:
            logger.warning(
                "No checkpoint path provided. Initializing parameters with random noise."
            )
            init_state = self.model.initialize_parameters_recursively(rng_key)

        self.axlearn_state = nnx.Param(init_state)

    def get_mrope_input_positions(
            self, prompt_token_ids: List[int],
            mm_features: List[Any]) -> Tuple[jax.Array, int]:
        seq_len = len(prompt_token_ids)
        positions = jnp.vstack([jnp.arange(seq_len, dtype=jnp.int32)] * 3)
        return positions, 0
