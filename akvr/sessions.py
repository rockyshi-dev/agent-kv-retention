"""Build the sessions offline (CPU only) from a vLLM source checkout and the model's tokenizer.

Every session is a chat rendered with the model's own template: a system text and an earlier
user turn cut from the vLLM source tree, ending with the assistant header, of exactly N - 768
tokens; and a 256-token continuation that closes the assistant turn and adds a user message
(the tool result). The 512 generated tokens are appended at run time, exactly as returned.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile

from .spec import (CLOSING, COMMIT, LEVELS, MODEL, REFERENCE_IDS_SHA256, ROUNDS, SESSIONS_PER_LEVEL,
                   atomic_json, check_model, digest, ids_digest, prompt_length)

SCHEMA = 'agent-kv-retention-sessions-v1'


def source_material(root):
    """Concatenated text of every committed .py file under vllm/ at the pinned commit."""
    actual = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != COMMIT:
        raise ValueError(f'vLLM checkout is at {actual}, not {COMMIT}')
    # git archive reads committed blobs, so edits in the working tree do not matter.
    raw = subprocess.check_output(['git', '-C', str(root), 'archive', COMMIT, 'vllm'])
    files, texts, offset = [], [], 0
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        for item in sorted(archive.getmembers(), key=lambda x: x.name):
            if not item.isfile() or not item.name.endswith('.py'):
                continue
            data = archive.extractfile(item).read()
            text = data.decode('utf-8')
            files.append({'path': item.name, 'sha256': hashlib.sha256(data).hexdigest(),
                          'start': offset, 'end': offset + len(text)})
            offset += len(text)
            texts.append(text)
    return ''.join(texts), files


def encode(tok, text):
    return tok.encode(text, add_special_tokens=False)


def trim(text, target, count):
    """Longest-prefix search for a cut of `text` whose rendering has exactly `target` tokens."""
    lo, hi = 0, len(text)
    while lo <= hi:
        mid = (lo + hi) // 2
        size = count(text[:mid])
        if size == target:
            return text[:mid]
        if size < target:
            lo = mid + 1
        else:
            hi = mid - 1
    for end in range(max(0, lo - 128), min(len(text), lo + 128) + 1):
        if count(text[:end]) == target:
            return text[:end]
    raise ValueError('No exact source-text boundary; try another offset')


def render(tok, system, user, assistant):
    return tok.apply_chat_template([
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': user},
        {'role': 'assistant', 'content': assistant},
        {'role': 'user', 'content': 'Continue reviewing the source code.'},
    ], tokenize=False, add_generation_prompt=True)


def closing(tok, text):
    """The template's continuation after assistant content, through the next assistant header."""
    marker = 'ASSISTANT_BOUNDARY_764830'
    rendered = tok.apply_chat_template([
        {'role': 'assistant', 'content': marker},
        {'role': 'user', 'content': text},
    ], tokenize=False, add_generation_prompt=True)
    if rendered.count(marker) != 1:
        raise ValueError('Template did not preserve the assistant boundary')
    return rendered.split(marker)[1]


def build(tok, source, files, rounds=ROUNDS, levels=LEVELS):
    sessions, first_blocks = [], set()
    special = set(tok.all_special_ids)
    # Distinct early system text gives each session its own first cache block.
    # Skip short files (mostly licence headers); every byte comes from a committed blob.
    positions = [f['start'] + 500 for f in files if f['end'] - f['start'] >= 2500]
    cursor = 0
    for round_id in range(rounds):
        for n in levels:
            for index in range(SESSIONS_PER_LEVEL):
                for _ in range(len(positions)):
                    offset = positions[cursor % len(positions)]
                    cursor += 1
                    chunk = source[offset:offset + 200000]
                    system = chunk[:160]
                    earlier = chunk[160:600]
                    if tuple(encode(tok, render(tok, system, '', earlier))[:16]) in first_blocks:
                        continue
                    try:
                        body = trim(chunk[600:], prompt_length(n),
                                    lambda s: len(encode(tok, render(tok, system, s, earlier))))
                        rendered = render(tok, system, body, earlier)
                        ids = encode(tok, rendered)
                        block = tuple(ids[:16])
                        if len(ids) != prompt_length(n) or block in first_blocks:
                            continue
                        if any(special.intersection(encode(tok, s)) for s in (system, body, earlier)):
                            continue
                        tool = trim(chunk[-10000:], CLOSING, lambda s: len(encode(tok, closing(tok, s))))
                        tail = closing(tok, tool)
                        tail_ids = encode(tok, tail)
                        if len(tail_ids) != CLOSING or special.intersection(encode(tok, tool)):
                            continue
                    except ValueError:
                        continue
                    first_blocks.add(block)
                    sessions.append({'id': f'r{round_id}-n{n}-s{index}', 'round': round_id,
                        'N': n, 'index': index, 'warmup': index == 0, 'source_offset': offset,
                        'system': system, 'earlier_user': body, 'earlier_assistant': earlier,
                        'prompt_text': rendered, 'prompt_ids': ids, 'tool_text': tool,
                        'closing_text': tail, 'closing_ids': tail_ids,
                        'text_sha256': hashlib.sha256((rendered + tail).encode()).hexdigest()})
                    break
                else:
                    raise RuntimeError(f'Cannot construct a distinct session r{round_id} n{n} s{index}')
    # Ordinary vocabulary ids for the filler, taken from the source text.
    vocabulary = sorted(set(encode(tok, source[:200000])) - special)
    if len(vocabulary) < 256:
        raise ValueError('Too few ordinary tokens for the filler')
    return {'schema': SCHEMA, 'model': MODEL, 'source_commit': COMMIT, 'source_files': files,
            'sessions': sessions, 'filler_vocabulary': vocabulary}


def validate(data, rounds=ROUNDS, levels=LEVELS):
    if (data['source_commit'], data['model']) != (COMMIT, MODEL):
        raise ValueError('Sessions were built for another model or source commit')
    expected = {(r, n, i) for r in range(rounds) for n in levels for i in range(SESSIONS_PER_LEVEL)}
    keys, blocks = set(), set()
    for s in data['sessions']:
        key = (s['round'], s['N'], s['index'])
        block = tuple(s['prompt_ids'][:16])
        if key in keys or block in blocks:
            raise ValueError('Duplicate session or first cache block')
        if len(s['prompt_ids']) != prompt_length(s['N']) or len(s['closing_ids']) != CLOSING:
            raise ValueError('Incorrect offline lengths')
        if s['warmup'] != (s['index'] == 0):
            raise ValueError('Session 0 of every level is the warm-up')
        keys.add(key)
        blocks.add(block)
    if keys != expected:
        raise ValueError('Incomplete session set')


def main(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    os.environ.setdefault('USE_TORCH', '0')
    os.environ.setdefault('USE_TF', '0')
    model = check_model(args.model, args.allow_other_model)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    text, files = source_material(args.vllm_source)
    result = build(tok, text, files)
    result['revision'] = model['revision']
    result['model_reference'] = model['reference']
    result['tokenizer_files'] = {k: v for k, v in model['files_sha256'].items()
                                 if k in ('tokenizer.json', 'tokenizer_config.json', 'config.json')}
    validate(result)
    result['ids_sha256'] = ids_digest(result['sessions'])
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, result)
    print(json.dumps({'path': str(output), 'sha256': digest(output), 'sessions': len(result['sessions']),
                      'ids_sha256': result['ids_sha256'],
                      'reference_sessions': result['ids_sha256'] == REFERENCE_IDS_SHA256}))
