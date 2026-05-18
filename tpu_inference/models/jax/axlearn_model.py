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

from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
from axlearn.common.checkpointer import CheckpointValidationType
from axlearn.common.module import functional as F
from axlearn.common.state_builder import (Builder,
                                          TensorStoreStateStorageBuilder)
from axlearn.experiments.text.gpt.c4_trainer import named_trainer_configs
from axlearn.experiments.text.gpt.fuji import model_config as fuji_model_config
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


class AxLearnForCausalLM(nnx.Module):

    def __init__(self, vllm_config: VllmConfig, rng_key: jax.Array,
                 mesh: Mesh) -> None:
        self.vllm_config = vllm_config

        self.mesh = mesh

        model_config_hf = vllm_config.model_config.hf_config
        self.hidden_dim = model_config_hf.hidden_size
        self.vocab_size = model_config_hf.vocab_size
        axlearn_cfg = getattr(vllm_config, "additional_config",
                              {}).get("axlearn_config", {})
        model_name = axlearn_cfg.get("model_name", None)

        configs_map = named_trainer_configs()
        if model_name and model_name in configs_map:
            logger.info(
                f"Instantiating model structure directly from AxLearn registry: {model_name}"
            )
            trainer_cfg = configs_map[model_name]()
            self.fuji_config = trainer_cfg.model.set(name="fuji")
        else:
            logger.info(
                f"Named config '{model_name}' not found in AxLearn registry. Mapping properties from HF config."
            )
            self.fuji_config = fuji_model_config(
                num_layers=model_config_hf.num_hidden_layers,
                num_heads=model_config_hf.num_attention_heads,
                hidden_dim=model_config_hf.hidden_size,
                num_kv_heads=model_config_hf.num_key_value_heads,
                vocab_size=model_config_hf.vocab_size,
                rope_theta=getattr(model_config_hf, "rope_theta", 10000.0),
                shared_lm_head=getattr(model_config_hf, "tie_word_embeddings",
                                       False),
                dropout_rate=0.0,
                ffn_dim=model_config_hf.intermediate_size,
                flash_attention=getattr(model_config_hf,
                                        "use_flash_attention_2", False),
                stack_cfg=None,
                pad_token_id=model_config_hf.pad_token_id,
                eos_token_id=model_config_hf.eos_token_id,
            ).set(name="fuji")

        self.model = self.fuji_config.instantiate(parent=None)

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

        input_ids_2d = jnp.expand_dims(input_ids, axis=1)
        positions_2d = jnp.expand_dims(attention_metadata.input_positions,
                                       axis=1)

        input_batch = dict(
            input_ids=input_ids_2d,
            input_positions=positions_2d,
            input_segment_ids=jnp.ones_like(input_ids_2d),
        )

        inputs = dict(
            input_batch=input_batch,
            return_aux=True,
        )

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
