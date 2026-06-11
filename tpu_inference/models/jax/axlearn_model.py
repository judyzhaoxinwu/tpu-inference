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


def _recursive_sanitize_specs(cfg, allowed_axes):
    from axlearn.common.config import ConfigBase
    if isinstance(cfg, ConfigBase):
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
                        _recursive_sanitize_specs(v, allowed_axes)
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
                        _recursive_sanitize_specs(item, allowed_axes)
                cfg.set(**{key: type(val)(new_list)})
            elif hasattr(val, 'klass') or isinstance(val, ConfigBase):
                _recursive_sanitize_specs(val, allowed_axes)
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
            from axlearn.common.attention import (FusedGroupedQKVLinear,
                                                  FusedQKVLinear,
                                                  GroupedQueryAttention,
                                                  MultiheadAttention,
                                                  RoFormerQKVLinear)
            from axlearn.common.layers import RMSNorm
            from axlearn.experiments.text.gpt.c4_trainer import \
                named_trainer_configs as c4_configs
            from axlearn.experiments.text.gpt.common import \
                model_config as common_model_config
            from axlearn.experiments.text.gpt.pajama_trainer import \
                named_trainer_configs as pajama_configs

        model_config_hf = vllm_config.model_config.hf_config
        if (hasattr(model_config_hf, "thinker_config")
                and hasattr(model_config_hf.thinker_config, "text_config")):
            model_config_hf = model_config_hf.thinker_config.text_config
        elif hasattr(model_config_hf, "text_config"):
            model_config_hf = model_config_hf.text_config

        self.hidden_dim = getattr(model_config_hf, "hidden_size",
                                  getattr(model_config_hf, "hidden_dim", None))
        self.vocab_size = getattr(model_config_hf, "vocab_size", None)
        axlearn_cfg = getattr(vllm_config, "additional_config",
                              {}).get("axlearn_config", {})
        model_name = axlearn_cfg.get("model_name", None)

        configs_map = {}
        configs_map.update(c4_configs())
        configs_map.update(pajama_configs())
        if model_name and model_name in configs_map:
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
                atten_cfg = GroupedQueryAttention.default_config().set(
                    hidden_dim=atten_hidden_dim)
                atten_input_linear = FusedGroupedQKVLinear.default_config(
                ).set(num_kv_heads=num_kv_heads)
            else:
                atten_cfg = MultiheadAttention.default_config().set(
                    hidden_dim=atten_hidden_dim)
                atten_input_linear = FusedQKVLinear.default_config()

            # 2. Setup Rotary Position Embeddings (RoPE)
            attention_qkv_linear = RoFormerQKVLinear.default_config().set(
                input_linear=atten_input_linear,
                rotary_value=False,
            )
            rope_theta = getattr(model_config_hf, "rope_theta", 10000.0)
            attention_qkv_linear.rope_pos_emb_layer.set(theta=rope_theta)

            # 3. Setup MoE parameters dynamically (if expert keys are present)
            num_experts = getattr(
                model_config_hf, "num_local_experts",
                getattr(model_config_hf, "num_experts", None))
            ffn_layer_types = None
            expert_cfg = None
            if num_experts is not None:
                from axlearn.common.mixture_of_experts import MixtureOfExperts
                ffn_layer_types = ["dense", "sparse"]
                expert_cfg = MixtureOfExperts.default_config().set(
                    num_experts=num_experts)

            self.axlearn_model_config = common_model_config(
                num_layers=model_config_hf.num_hidden_layers,
                hidden_dim=model_config_hf.hidden_size,
                num_heads=model_config_hf.num_attention_heads,
                vocab_size=model_config_hf.vocab_size,
                activation_fn=("nn.silu", "linear"),  # SwiGLU
                ffn_dim=model_config_hf.intermediate_size,
                normalization=RMSNorm.default_config().set(
                    eps=getattr(model_config_hf, "rms_norm_eps", 1e-6),
                    forward_dtype=jnp.float32,
                ),
                attention_cfg=atten_cfg,
                attention_qkv_linear=attention_qkv_linear,
                ffn_layer_types=ffn_layer_types,
                expert_cfg=expert_cfg,
                pad_token_id=model_config_hf.pad_token_id,
                eos_token_id=model_config_hf.eos_token_id,
            ).set(name=model_name or "axlearn_model")

        abstract_mesh = jax.sharding.get_abstract_mesh()
        allowed_axes = abstract_mesh.axis_names if not abstract_mesh.empty else (
            "model", "expert")
        logger.info(
            f"Sanitizing model PartitionSpecs to match active compilation mesh axes: {allowed_axes}"
        )
        _recursive_sanitize_specs(self.axlearn_model_config, allowed_axes)
        _recursive_set_block_size(self.axlearn_model_config)

        # Dynamically calculate MoE outer batch size from active serving mesh
        outer_batch_size = 1
        for axis in ("data", "fsdp"):
            if axis in self.mesh.shape:
                outer_batch_size *= self.mesh.shape[axis]
        _recursive_set_outer_batch(self.axlearn_model_config, outer_batch_size)

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
        return kv_caches, hidden_states, [], None

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
