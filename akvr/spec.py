"""Fixed parameters of the measurement. This module never imports CUDA libraries."""
import hashlib
import json
import os
from pathlib import Path

COMMIT = 'ced6857afa0ea7b2e3f0846a62e1394e90f15607'  # vLLM v0.30.0
VLLM_VERSION = '0.30.0+cu129'
TORCH_CUDA = '12.9'
MODEL = 'Qwen/Qwen2.5-7B-Instruct'
REFERENCE_REVISION = 'a09a35458c702b33eeacc393d103063234e8bc28'
LEVELS = (2048, 8192, 28672)
ARMS = ('KEEP', 'DROP', 'NATIVE', 'NATIVE-ALL', 'LAZY')
OFFLOAD_ARMS = ARMS[2:]
# One server start per group; KEEP and DROP share the default server.
GROUPS = ('DEFAULT', 'NATIVE', 'NATIVE-ALL', 'LAZY')
ROUNDS = 5
SESSIONS_PER_LEVEL = 5          # one warm-up and four measured sessions
GENERATED = 512                 # tokens generated in turn 1
CLOSING = 256                   # tokens that close the turn and add a tool result
RESUME_OUTPUT = 8
FILLER_PROMPT = 4096
FILLER_FACTOR = 1.5
RESUME_WAIT_S = 2
BYTES_PER_TOKEN = 57344         # 2 x 28 layers x 4 KV heads x 128 x 2 bytes
HOST_BYTES = 17179869184        # 16 GiB host buffer of every offload arm
RATIO_THRESHOLD = 0.8
QUIET_ATTEMPTS = 3              # a round and at most two reruns
TRACE_SHA = '9d265eae69a31cae203848bea936f018148eed7ca8bf56050c5abe96da0b4e6b'
TRACE_REPO = 'uw-syfi/TraceLab'
TRACE_RELEASE = 'v0.0.1'
TRACE_SOURCE_COMMIT = 'fc061574a0f9700ccf88be33112adc8e86b425fa'
CONTEXT_LIMIT = 32768
# Content hash of the sessions built from the reference inputs (see README).
REFERENCE_IDS_SHA256 = 'a766a2cbf6d492a4c285356f1bc745183c328aa4d3e0e45e80a1a5a8eb5dc5c9'

PACKAGE = Path(__file__).resolve().parent


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temp.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')
    os.replace(temp, path)


def code_hashes():
    """sha256 of every module of this package; a start test binds the timed run to them."""
    return {p.name: digest(p) for p in sorted(PACKAGE.iterdir()) if p.suffix == '.py'}


def config(arm):
    """The --kv-transfer-config of an arm, or None for vLLM's defaults."""
    if arm in ('KEEP', 'DROP', 'DEFAULT'):
        return None
    if arm not in ARMS:
        raise ValueError(arm)
    extra = {'cpu_bytes_to_use': HOST_BYTES}
    if arm == 'NATIVE-ALL':
        extra['offload_prompt_only'] = False
    if arm == 'LAZY':
        extra['lazy_offload'] = True
    return {'kv_connector': 'SimpleCPUOffloadConnector' if arm == 'LAZY' else 'OffloadingConnector',
            'kv_role': 'kv_both', 'kv_connector_extra_config': extra}


def prompt_length(n):
    return n - GENERATED - CLOSING


def reusable(n):
    """C: full 16-token blocks whose KV exists after turn 1 (the last output has none)."""
    return ((prompt_length(n) + GENERATED - 1) // 16) * 16


def host_min(arm, n):
    """Least Hh that counts as a complete host copy (NATIVE stores prompt blocks only)."""
    return (prompt_length(n) // 16) * 16 - 16 if arm == 'NATIVE' else reusable(n) - 16


def ids_digest(sessions):
    rows = [[s['id'], s['prompt_ids'], s['closing_ids']] for s in sessions]
    return hashlib.sha256(json.dumps(rows, separators=(',', ':')).encode()).hexdigest()


# sha256 of the reference snapshot's files. generation_config.json sets the server's default
# sampling parameters (repetition_penalty 1.05, top_k 20, top_p 0.8), so it is pinned too.
REFERENCE_MODEL_FILES = {
    'config.json': '7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c',
    'generation_config.json': '3a8f9087e486054c8a4a08dae2e5a3ba62e23da212b5b8c08bc42cb983c3459f',
    'tokenizer.json': 'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539',
    'tokenizer_config.json': '5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583',
}


def check_model(directory, allow_other=False):
    """The model must be the reference snapshot (by file hashes) unless `allow_other` is given,
    and in any case have the KV geometry and context the arithmetic assumes."""
    directory = Path(directory)
    missing = [name for name in REFERENCE_MODEL_FILES if not (directory / name).is_file()]
    if missing:
        raise ValueError(f'Model directory lacks {", ".join(missing)}')
    cfg = json.loads((directory / 'config.json').read_text())
    if cfg.get('torch_dtype', cfg.get('dtype')) != 'bfloat16':
        raise ValueError('The default model dtype must be bfloat16')
    if cfg.get('max_position_embeddings') != CONTEXT_LIMIT:
        raise ValueError('The model context must be 32768 tokens')
    kv = (2 * cfg['num_hidden_layers'] * cfg['num_key_value_heads']
          * (cfg['hidden_size'] // cfg['num_attention_heads']) * 2)
    if kv != BYTES_PER_TOKEN:
        raise ValueError(f'KV cache bytes per token is {kv}, not {BYTES_PER_TOKEN}')
    files = {name: digest(directory / name) for name in REFERENCE_MODEL_FILES}
    different = sorted(name for name, sha in files.items() if REFERENCE_MODEL_FILES[name] != sha)
    if different and not allow_other:
        raise ValueError('Not the reference model snapshot (different ' + ', '.join(different)
                         + '); pass --allow-other-model to use it anyway')
    resolved = directory.resolve()
    revision = resolved.name if resolved.parent.name == 'snapshots' else None
    return {'path': str(directory), 'resolved': str(resolved), 'revision': revision, 'files_sha256': files,
            'reference': not different, 'different_files': different, 'allow_other_model': bool(allow_other)}
