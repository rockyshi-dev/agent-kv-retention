"""Synthetic records shared by the tests."""
from pathlib import Path

from akvr.spec import ARMS, LEVELS, ROUNDS, prompt_length, reusable

FIXTURES = Path(__file__).resolve().parent / 'fixtures'


def fixture(name):
    return (FIXTURES / name).read_text()


def metric_text(success=0, queries=0, hits=0, external=0, load=None, running=0, waiting=0):
    rows = [f'vllm:{name}_total{{model_name="m",engine="0"}} {v}' for name, v in (
        ('request_success', success), ('prefix_cache_queries', queries), ('prefix_cache_hits', hits),
        ('external_prefix_cache_hits', external))]
    if load is not None:
        rows.append(f'vllm:kv_offload_load_bytes_total{{model_name="m",engine="0"}} {load}')
    rows += [f'vllm:num_requests_running {running}', f'vllm:num_requests_waiting {waiting}']
    return '\n'.join(rows) + '\n'


def expected_counts(arm, n):
    g = reusable(n) if arm == 'KEEP' else 0
    h = prompt_length(n) if arm == 'NATIVE' else reusable(n) if arm in ('NATIVE-ALL', 'LAZY') else 0
    return g, h


def resume(n, g, h, ttft_ms, load=None, raw=False):
    """A resume record; with raw=True it also carries the /metrics text and stream events."""
    record = {'usage': {'prompt_tokens': n}, 'ttft_ms': ttft_ms,
              'delta': {'request_success': 1, 'prefix_cache_hits': g, 'external_prefix_cache_hits': h,
                        'kv_offload_load_bytes': load, 'prefix_cache_queries': n,
                        'num_requests_running': 0, 'num_requests_waiting': 0}}
    if raw:
        base = 1000
        before = metric_text(base, 10 * base, base, base, None if load is None else base)
        after = metric_text(base + 1, 10 * base + n, base + g, base + h, None if load is None else base + load)
        start = 5_000_000_000
        record.update(before={'raw': before}, after={'raw': after}, start_ns=start, events=[
            {'arrival_ns': start + int(ttft_ms * 1e6), 'chunk': {'choices': [{'text': '', 'token_ids': [7]}]}},
            {'arrival_ns': start + int(ttft_ms * 1e6) + 9_000_000, 'chunk': {'choices': [], 'usage': {'prompt_tokens': n}}}])
    return record


def bundle(ratio=.7, raw=False):
    """Five selected rounds, every arm and level with four measured sessions; DROP takes 100 ms."""
    rounds = []
    for r in range(ROUNDS):
        records = []
        for a in ARMS:
            for n in LEVELS:
                for i in range(1, 5):
                    g, h = expected_counts(a, n)
                    load = h * 57344 if a in ('NATIVE', 'NATIVE-ALL') else None
                    records.append({'id': f'r{r}-n{n}-s{i}', 'arm': a, 'N': n, 'round': r, 'index': i,
                                    'warmup': False, 'turn1': {'token_ids': [17] * 512},
                                    'resume': resume(n, g, h, 100 if a == 'DROP' else ratio * 100, load, raw)})
        rounds.append({'round': r, 'attempt': 0, 'selected': True, 'quiet': {'pass': True},
                       'group_order': ['DEFAULT', 'NATIVE', 'NATIVE-ALL', 'LAZY'],
                       'starts': {a: {'pass': True, 'pool_tokens': 80000, 'pool_bytes': 57344 * 80000}
                                  for a in ARMS},
                       'sessions': records})
    return {'rounds': rounds}


def telemetry_sample(t, total, busy, owned, clock=2520, foreign=()):
    return {'time': t, 'cpu': {'0': {'total': total, 'busy': busy}, '1': {'total': total, 'busy': busy}},
            'owned_ticks': {'0': owned, '1': owned}, 'placement_ok': True,
            'gpu': {'clock_mhz': clock, 'foreign_pids': list(foreign)}}
