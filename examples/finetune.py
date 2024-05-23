from typing import Any, Iterable, Tuple, Sequence, Dict
from absl import app, flags, logging
from functools import partial, reduce
from timeit import default_timer as timer
import operator
import os
import re
import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np
import orbax.checkpoint
import optax
import sentencepiece as spm
import tensorflow_datasets as tfds
from torch.utils import data
from jax.sharding import Mesh, PartitionSpec, NamedSharding

from gemma import params as params_lib
from gemma import sampler as sampler_lib
from gemma import transformer as transformer_lib

_PRETRAINED_CHECKPOINT_PATH = flags.DEFINE_string("check_path", "/opt/gemma/models/Flax/2b-it/2/2b-it", "Path to pretrained model weights and state.")
_VOCAB_PATH = flags.DEFINE_string("vocab_path", "/opt/gemma/models/Flax/2b-it/2/tokenizer.model", "Path to tokenizer model for tokenization.")
_BATCH_SIZE = flags.DEFINE_integer("batch_size", 4, "Batch size for finetuning.")
_SEQ_LEN = flags.DEFINE_integer("seq_len", 8 * 1024, "Sequence length for finetuning.")
_LEARNING_RATE = flags.DEFINE_float("lr", 1e-4, "Learning rate for finetuning.")
_LOG_FREQ = flags.DEFINE_integer("log_freq", 32, "Loss loggging frequency in number of iterations.")
_EVAL_FREQ = flags.DEFINE_integer("eval_freq", 96, "Evaluation frequency in number of iterations.")
_EPOCHS = flags.DEFINE_integer("epochs", 5, "Evaluation frequency in number of iterations.")
_BENCHMARK = flags.DEFINE_boolean("benchmark", False, "Whether or not to benchmark computation.")
_MULTIDEVICE = flags.DEFINE_boolean("multi_device", False, "Whether or not to use .")

ParameterPytree = Dict[str, Any]
GEMMA_PARTITION_RULES = [
    # Embedding table
    ("embedder", 
        (
            (("input_embedding",), PartitionSpec(None, None)),
        )
    ),
    # Final Norm
    ("final_norm", 
        (
            (("scale",), PartitionSpec(None)),
        )
    ),
    # Attention blocks
    (r"^layer_\d+$", 
        (
            # Layernorms
            (("pre_attention_norm", "scale"), PartitionSpec(None)),
            (("pre_ffw_norm", "scale"), PartitionSpec(None)),

            # Attn layer
            (("attn", "q_einsum"), PartitionSpec('TP', None, None)),
            (("attn", "kv_einsum"), PartitionSpec(None, None, None, None)),
            (("attn", "attn_vec_einsum"), PartitionSpec('TP', None, None)),

            # MLP layer
            (("mlp", "gating_einsum"), PartitionSpec(None, None, 'TP')),
            (("mlp", "linear"), PartitionSpec('TP', None)),            
        )
    ),
]

def shard_parameters(params: ParameterPytree, mesh: Mesh) -> ParameterPytree:
    def partition_dict_set(part_dict : ParameterPytree, layer: str, leaf_path: Iterable[str], sharding: NamedSharding) -> None:
        key_path = ["params", layer] + list(leaf_path)
        for key in key_path[:-1]:
            part_dict = part_dict.setdefault(key, {})       
        part_dict[key_path[-1]] = sharding
    
    partition_dict = {}
    for layer in params['params']:
        for pattern, rules in GEMMA_PARTITION_RULES:
            if re.match(pattern, layer):
                for rule in rules:
                    leaf_path, spec = rule
                    if leaf_path[0] in params['params'][layer].keys():
                        partition_dict_set(partition_dict, layer, leaf_path, NamedSharding(mesh, spec))

    partitioned_parameters = jax.tree_util.tree_map_with_path(lambda keypath, leaf: 
                                                              jax.device_put(leaf, reduce(operator.getitem, [obj.key for obj in keypath], partition_dict)), 
                                                              params)
    return partitioned_parameters



class EtoFTokenizer:
    def __init__(self, spm_processor: spm.SentencePieceProcessor):
        self.spm_processor = spm_processor
        self.tokenize_source = partial(self._tokenize, 
                                       prefix='Translate this into French:\n', 
                                       suffix='\n', 
                                       add_eos=False)
        self.tokenize_dst = partial(self._tokenize, prefix='', suffix='', add_eos=True)
    
    @property
    def pad_id(self) -> int:
        return self.spm_processor.pad_id()

    def to_string(self, tokens: jax.Array) -> str:
        """Convert an array of tokens to a string."""
        return self._spm_processor.EncodeIds(tokens.tolist())
    
    def _tokenize(
        self,
        example: str | bytes,
        prefix: str = '',
        suffix: str = '',
        add_eos: bool = True,
    ) -> jax.Array:
        """
        Tokenization function.

        Args:
            example: input string to tokenize.
            prefix:  prefix to add to the input string.
            suffix:  suffix to add to the input string.
            add_eos: if True, add an end of sentence token at the end of the output
                    sequence.
        Returns:
            Tokens corresponding to the input string.
        """
        if isinstance(example, bytes):
            example = example.decode("utf-8")
        int_list = [self.spm_processor.bos_id()]
        try:
            int_list.extend(self.spm_processor.EncodeAsIds(prefix + example + suffix))
        except:
            breakpoint()
        if add_eos:
            int_list.append(self.spm_processor.eos_id())

        return jnp.array(int_list, dtype=jnp.int32)

class EtoFMTNTDataset(data.Dataset):
    def __init__(self, tokenizer: EtoFTokenizer, max_seq_len: int, mode: str):
        assert mode == 'train' or mode =='valid'
        self._tokenizer = tokenizer
        self._base_data = list(tfds.load("mtnt/en-fr", split=mode).as_numpy_iterator())
        self._max_seq_len = max_seq_len

    def _pad_upto_max_len(self, input_tensor: jax.Array, pad_value: int | bool) -> jax.Array:
        seq_len = input_tensor.shape[0]
        pad_len = max(self._max_seq_len - seq_len, 0)
        return jnp.pad(input_tensor, [[0, pad_len]], mode='constant', constant_values=pad_value)
    
    def _create_input(self, src_tokens: jax.Array, dst_tokens: jax.Array) -> Tuple[jax.Array, jax.Array]:
        tokens = jnp.concatenate([src_tokens, dst_tokens])
        src_mask = jnp.zeros_like(src_tokens, dtype=jnp.bool)
        dst_mask = jnp.ones_like(dst_tokens, dtype=jnp.bool)
        mask = jnp.concatenate([src_mask, dst_mask])
        tokens = self._pad_upto_max_len(tokens, self._tokenizer.pad_id)
        mask = self._pad_upto_max_len(mask, False)
        return tokens, mask
    
    def __len__(self) -> int:
        return len(self._base_data)

    def __getitem__(self, index) -> Tuple[jax.Array, jax.Array]:
        src, dst = self._base_data[index]['src'], self._base_data[index]['dst']
        src_tokens, dst_tokens = self._tokenizer.tokenize_source(src), self._tokenizer.tokenize_dst(dst)
        tokens, target_mask = self._create_input(src_tokens, dst_tokens)
        return tokens, target_mask
    
    @staticmethod
    def collator(batch):
        per_example_tokens, per_example_masks = zip(*batch)
        return jnp.stack(per_example_tokens), jnp.stack(per_example_masks)

def load_model_params(checkpoint: str) -> Tuple[transformer_lib.TransformerConfig, ParameterPytree]:
    if checkpoint is None:
        raise ValueError("Absolute path to checkpoint must not be None")
    
    logging.info(f"Searching for checkpoint in {checkpoint}.")
    params = params_lib.load_and_format_params(checkpoint)
    config = transformer_lib.TransformerConfig.from_params(params, cache_size=1024)
    model = transformer_lib.Transformer(config=config)
    return config, model, {"params" : params["transformer"]}

def get_positions_and_attention_mask(input: jax.Array, pad_id: int) -> jax.Array:
    pad_mask = input != pad_id
    positions = transformer_lib.build_positions_from_mask(pad_mask)
    attention_mask = transformer_lib.make_causal_attn_mask(pad_mask)
    return positions, attention_mask

def model_forward_and_loss(
      model: transformer_lib.Transformer,
      params: ParameterPytree,
      tokens: jax.Array,
      input_mask: jax.Array,
      positions: jax.Array,
      attention_mask: jax.Array
) -> jax.Array:
   logits, _ = model.apply(params, tokens, positions, None, attention_mask)
   logits = logits[:, :-1, :]
   target_tokens = tokens[:, 1:]
   target_mask = input_mask[:, 1:]
   one_hot = jax.nn.one_hot(target_tokens, logits.shape[-1])
   one_hot = one_hot * target_mask.astype(one_hot.dtype)[..., None]
   norm_factor = 1 / (jnp.sum(target_mask, axis=-1) + 1e-8)
   per_example_loss = -jnp.sum((jax.nn.log_softmax(logits, axis=-1) * one_hot), axis=(-1, -2)) * norm_factor
   return jnp.mean(per_example_loss)

def train_step(
      model: transformer_lib.Transformer, 
      params: ParameterPytree, 
      optimizer: optax.GradientTransformation, 
      opt_state: optax.OptState, 
      tokens: jax.Array, 
      input_mask: jax.Array,
      pad_id: int
) -> Tuple[jax.Array, ParameterPytree, optax.OptState]:
   positions, attention_mask = get_positions_and_attention_mask(tokens, pad_id)
   loss, grads = jax.value_and_grad(model_forward_and_loss, argnums=1)(model, params, tokens, input_mask, positions, attention_mask)
   updates, opt_state = optimizer.update(grads, opt_state)
   params = optax.apply_updates(params, updates)
   return loss, params, opt_state

def eval_step(
      model: transformer_lib.Transformer,
      params: ParameterPytree,
      tokens: jax.Array,
      input_mask: jax.Array,
      pad_id: int
) -> jax.Array:
   positions, attention_mask = get_positions_and_attention_mask(tokens, pad_id)
   return model_forward_and_loss(model, params, tokens, input_mask, positions, attention_mask)

def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
       raise app.UsageError("Too many command-line arguments.")
    
    jax.distributed.initialize()

    # Init / load model parameters
    config, model, params = load_model_params(_PRETRAINED_CHECKPOINT_PATH.value)
    print(f"Process {jax.process_index()} on node {os.environ.get('SLURM_NODEID')}")

    # Shard model parameters
    if _MULTIDEVICE.value:
        num_local_devices = jax.local_device_count()
        assert config.num_heads % num_local_devices == 0, "number of available device must be evenly divide number of attention heads"
        devices = np.asarray(jax.devices()).reshape(-1, num_local_devices)
        mesh = Mesh(devices, ('DP', 'TP'))
        params = shard_parameters(params, mesh)


if __name__ == "__main__":
  app.run(main)



