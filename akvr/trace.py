"""Count tool calls in the TraceLab coding-agent trace whose idle context fits one 24 GB card.

The idle context of a tool record is the input plus output tokens of the LLM round that issued
the call, as counted by the trace's providers (not re-tokenized for Qwen).
"""
import collections
import gzip
import json
import math
from pathlib import Path

from .spec import (BYTES_PER_TOKEN, CONTEXT_LIMIT, TRACE_RELEASE, TRACE_REPO, TRACE_SHA, TRACE_SOURCE_COMMIT,
                   atomic_json, digest)

KS = (1, 2, 4, 8)


def count(path, expected_sha=TRACE_SHA):
    if expected_sha is not None and digest(path) != expected_sha:
        raise ValueError(f'{path} is not the TraceLab {TRACE_RELEASE} release file')
    histogram, long_histogram, unknown = collections.Counter(), collections.Counter(), collections.Counter()
    rounds = 0
    with gzip.open(path, 'rt') as f:
        for line_number, line in enumerate(f, 1):
            r = json.loads(line)
            rounds += 1
            if not isinstance(r.get('tools'), list):
                raise ValueError(f'Invalid tools field at row {line_number}')
            if not r['tools']:
                continue
            for key in ('input_tokens_total', 'output_tokens'):
                if type(r.get(key)) is not int or r[key] < 0:
                    raise ValueError(f'Invalid {key} at row {line_number}')
            n = r['input_tokens_total'] + r['output_tokens']
            for tool in r['tools']:
                histogram[n] += 1
                duration = tool.get('tool_wall_latency_ms')
                if duration is None:
                    unknown[n] += 1
                elif type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0:
                    raise ValueError(f'Invalid duration at row {line_number}')
                elif duration > 1000:
                    long_histogram[n] += 1
    return {'rounds': rounds, 'tools': sum(histogram.values()), 'histogram': dict(histogram),
            'long_known_histogram': dict(long_histogram), 'unknown_duration_histogram': dict(unknown)}


def _capacity(hist, denominator, pool, extra=None):
    out = {}
    for k in KS:
        exceeds = sum(v for n, v in hist.items() if n * BYTES_PER_TOKEN * k > pool)
        out[str(k)] = {'exceeds': exceeds, 'denominator': denominator,
                       'share': exceeds / denominator if denominator else None,
                       'keep_pool_bytes': pool, **(extra or {})}
    return out


def report(data, pools=None):
    """Shares from the histograms; capacity shares need the KEEP arm's logged pool in bytes."""
    hist = {int(k): v for k, v in data['histogram'].items()}
    within = {n: v for n, v in hist.items() if n <= CONTEXT_LIMIT}
    eligible = sum(within.values())
    total = sum(hist.values())
    long = {int(n): v for n, v in data['long_known_histogram'].items() if int(n) <= CONTEXT_LIMIT}
    denom = sum(long.values())
    missing = sum(data['unknown_duration_histogram'].values())
    missing_within = sum(v for n, v in data['unknown_duration_histogram'].items() if int(n) <= CONTEXT_LIMIT)
    result = {'scope': 'Provider input_tokens_total plus output_tokens; not Qwen-retokenized lengths.',
              'repo': TRACE_REPO, 'release': TRACE_RELEASE, 'sha256': TRACE_SHA,
              'source_commit': TRACE_SOURCE_COMMIT,
              'fields': {'input': 'input_tokens_total', 'output': 'output_tokens',
                         'duration': 'tools[].tool_wall_latency_ms'},
              'total_tools': total, 'within_32768': eligible,
              'within_32768_share': eligible / total if total else None,
              'missing_duration_tools': missing,
              'missing_duration_within_32768': missing_within,
              'over_1s_subgroup': {
                  'criterion': 'Known tools[].tool_wall_latency_ms > 1000',
                  'missing_duration_treatment': 'Included overall, excluded only from the >1 s subgroup; no imputation.',
                  'total_tools': sum(data['long_known_histogram'].values()),
                  'within_32768': denom,
                  'excluded_missing_duration_tools': missing,
                  'excluded_missing_duration_within_32768': missing_within},
              'capacity': None, 'over_1s_capacity': None}
    if pools is not None:
        if not isinstance(pools, dict) or not pools:
            raise ValueError('No logged KEEP pools supplied')
        for pool in pools.values():
            if type(pool) is not int or pool <= 0:
                raise ValueError('A logged KEEP pool in integer bytes is required')
        # Each logged pool is reported on its own; none is averaged or invented.
        result['capacity'] = {label: _capacity(within, eligible, pool) for label, pool in pools.items()}
        result['over_1s_capacity'] = {label: _capacity(
            long, denom, pool, {'excluded_missing_duration_within_32768': missing_within})
            for label, pool in pools.items()}
    result['capacity_pending_keep_pool'] = pools is None
    return result


def main(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    data = count(args.trace)
    pools = None
    if args.pools:
        source = json.loads(Path(args.pools).read_text())
        pools = source['trace_pool_bytes_by_round']
        pools = {f'round {k}': v for k, v in pools.items()}
    output.mkdir(parents=True)
    atomic_json(output / 'histogram.json', data)
    result = report(data, pools)
    atomic_json(output / 'report.json', result)
    print(json.dumps(result, indent=2))
    return 0
