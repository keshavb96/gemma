from typing import Any, Iterable, Tuple, Sequence, Dict
from absl import app, flags, logging
from functools import partial, reduce
import operator
import os
import re
import jax
import jax.numpy as jnp
import numpy as np
import optax
import sentencepiece as spm
import tensorflow_datasets as tfds
from torch.utils import data
from jax.sharding import Mesh, PartitionSpec, NamedSharding

from gemma import params as params_lib
from gemma import transformer as transformer_lib

_PRETRAINED_CHECKPOINT_PATH = flags.DEFINE_string("check_path", "/tmp/models/gemma/Flax/2b-it/2/2b-it", "Path to pretrained model weights and state.")
_VOCAB_PATH = flags.DEFINE_string("vocab_path", "/tmp/models/gemma/Flax/2b-it/2/tokenizer.model", "Path to tokenizer model for tokenization.")
_DATASET_DIR = flags.DEFINE_string("data_dir", "/tmp/datasets", "Path to datasets directory")
_BATCH_SIZE = flags.DEFINE_integer("batch_size", 4, "Batch size for finetuning.")
_SEQ_LEN = flags.DEFINE_integer("seq_len", 8 * 1024, "Sequence length for finetuning.")
_LEARNING_RATE = flags.DEFINE_float("lr", 1e-4, "Learning rate for finetuning.")
_LOG_FREQ = flags.DEFINE_integer("log_freq", 64, "Loss loggging frequency in number of iterations.")
_EVAL_FREQ = flags.DEFINE_integer("eval_freq", 256, "Evaluation frequency in number of iterations.")
_EPOCHS = flags.DEFINE_integer("epochs", 1, "Evaluation frequency in number of iterations.")
_JAX_CACHE = flags.DEFINE_string("jax_cache", None, "Path to jax persistent cache on shared filesystem between all processes.")
_JAX_SHARE_BINARY_BETWEEN_HOSTS = flags.DEFINE_boolean("share_binary", False, "Whether to share binary between hosts instead of every host compiling.")
_JAX_CACHE_ONLY = flags.DEFINE_boolean("cache_only", False, "Whether to just cache by doing a dummy computation or proceed with training after compilation.")

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
            (("attn", "q_einsum", "w"), PartitionSpec('TP', None, None)),
            (("attn", "kv_einsum", "w"), PartitionSpec(None, None, None, None)),
            (("attn", "attn_vec_einsum", "w"), PartitionSpec('TP', None, None)),

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
    
    def create_global_sharded_array(leaf: jax.Array, keypath: Tuple[str], partition_dict) -> jax.Array:
        sharding = reduce(operator.getitem, [obj.key for obj in keypath], partition_dict)
        return jax.make_array_from_process_local_data(sharding, leaf, leaf.shape)

    partition_dict = {}
    for layer in params['params']:
        for pattern, rules in GEMMA_PARTITION_RULES:
            if re.match(pattern, layer):
                for rule in rules:
                    leaf_path, spec = rule
                    if leaf_path[0] in params['params'][layer].keys():
                        partition_dict_set(partition_dict, layer, leaf_path, NamedSharding(mesh, spec))


    partitioned_parameters = jax.tree_util.tree_map_with_path(lambda keypath, leaf:
                                                              create_global_sharded_array(leaf, keypath, partition_dict),
                                                              params)
    return partitioned_parameters

def shard_inputs(*inputs: Tuple[jax.Array], 
                 global_input_shapes: Tuple[Tuple[int]], 
                 input_shardings: Tuple[NamedSharding]
) -> Tuple[jax.Array]:
    partitioned_inputs = tuple(jax.make_array_from_process_local_data(input_shardings[i], 
                                                                      input, 
                                                                      global_input_shapes[i]) 
                                                                      for i, input in enumerate(inputs))
    return partitioned_inputs

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
        self._base_data = list(tfds.load("mtnt/en-fr", data_dir=_DATASET_DIR.value, split=mode).as_numpy_iterator())
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
    
    jax.config.update("jax_compilation_cache_dir", _JAX_CACHE.value)
    jax.config.update("jax_share_binary_between_hosts", _JAX_SHARE_BINARY_BETWEEN_HOSTS.value)

    jax.distributed.initialize()
    global_rank = jax.process_index()

    # Init / load model parameters
    config, model, params = load_model_params(_PRETRAINED_CHECKPOINT_PATH.value)

    # Shard model parameters
    dp_dim = int(os.environ.get("SLURM_NNODES")) # Scheduler dependent (assumed SLURM here)
    tp_dim = len(jax.devices()) // dp_dim
    assert config.num_heads % tp_dim == 0, "number of devices in the TP mesh dim must be evenly divide number of attention heads"
    devices = np.asarray(jax.devices()).reshape(dp_dim, tp_dim)
    mesh = Mesh(devices, ('DP', 'TP'))
    params = shard_parameters(params, mesh)

    # Load vocab, tokenizer, create datasets and distributed dataloader
    dp_rank = int(os.environ.get("SLURM_NODEID")) # Scheduler dependent (assumed SLURM here)
    vocab = spm.SentencePieceProcessor()
    vocab.Load(_VOCAB_PATH.value)
    tokenizer = EtoFTokenizer(vocab)
    assert _BATCH_SIZE.value % dp_dim == 0, "batch size must be divisble by number data parallel dimension of the mesh"
    
    train_dataset =  EtoFMTNTDataset(tokenizer, max_seq_len=_SEQ_LEN.value, mode='train')
    valid_dataset = EtoFMTNTDataset(tokenizer, max_seq_len=_SEQ_LEN.value, mode='valid')
    sampler = lambda dataset: data.DistributedSampler(dataset, num_replicas=dp_dim, rank=dp_rank)

    train_dataloader = data.DataLoader(train_dataset, 
                                       sampler=sampler(train_dataset), 
                                       shuffle=False, 
                                       batch_size=_BATCH_SIZE.value // dp_dim, 
                                       collate_fn=EtoFMTNTDataset.collator
    )

    valid_dataloader = data.DataLoader(valid_dataset, 
                                       sampler=sampler(valid_dataset), 
                                       shuffle=False, 
                                       batch_size=_BATCH_SIZE.value // dp_dim, 
                                       collate_fn=EtoFMTNTDataset.collator
    )

    
    jitted_train_step = jax.jit(partial(train_step, model), static_argnames=['optimizer'])
    jitted_eval_step = jax.jit(partial(eval_step, model))

    optimizer = optax.adam(learning_rate=_LEARNING_RATE.value)
    opt_state = optimizer.init(params)
    
    # Barrier to ensure chronological printing
    jax.experimental.multihost_utils.sync_global_devices("all")
    
    avg_loss = 0
    n_steps = 0
    for epoch in range(_EPOCHS.value):
        train_dataloader.sampler.set_epoch(epoch)
        print(f"--------------- Rank {global_rank} starting epoch {epoch} ---------------")
        for i, (tokens, mask) in enumerate(train_dataloader):
            global_tokens, global_mask = shard_inputs(tokens, 
                                                      mask, 
                                                      global_input_shapes=((_BATCH_SIZE.value, _SEQ_LEN.value), (_BATCH_SIZE.value, _SEQ_LEN.value)), 
                                                      input_shardings=(NamedSharding(mesh, PartitionSpec("DP", None)), NamedSharding(mesh, PartitionSpec("DP", None))))

            train_loss, params, opt_state = jitted_train_step(params, optimizer, opt_state, global_tokens, global_mask, tokenizer.pad_id)

            n_steps += 1
            avg_loss += train_loss
            if (i + 1) % _LOG_FREQ.value == 0:
                # Only print statistics on rank 0
                if global_rank == 0:
                    print(f"Epoch {epoch + 1}, iteration {i + 1}: Average Loss = {avg_loss / n_steps}")
                
                avg_loss = 0
                n_steps = 0
            
            if (i + 1) % _EVAL_FREQ.value == 0:
                eval_loss = 0
                n_eval_steps = 0
                for _, (tokens, mask) in enumerate(valid_dataloader):
                    global_tokens, global_mask = shard_inputs(tokens, 
                                                              mask, 
                                                              global_input_shapes=((_BATCH_SIZE.value, _SEQ_LEN.value), (_BATCH_SIZE.value, _SEQ_LEN.value)), 
                                                              input_shardings=(NamedSharding(mesh, PartitionSpec("DP", None)), NamedSharding(mesh, PartitionSpec("DP", None))))                    
 
                    eval_loss += jitted_eval_step(params, global_tokens, global_mask, tokenizer.pad_id)
                    n_eval_steps += 1
                
                if global_rank == 0:
                    print(f"Average eval loss = {eval_loss / n_eval_steps}")


if __name__ == "__main__":
  app.run(main)




