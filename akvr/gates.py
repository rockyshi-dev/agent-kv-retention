"""Validity checks and the decision rule, as pure functions of the recorded data."""
import math
import re
import statistics

from .client import cache_config
from .spec import (ARMS, BYTES_PER_TOKEN, GENERATED, HOST_BYTES, LEVELS, OFFLOAD_ARMS, RATIO_THRESHOLD,
                   ROUNDS, SESSIONS_PER_LEVEL, host_min, reusable)

NATIVE_ARMS = ('NATIVE', 'NATIVE-ALL')
MEASURED = ROUNDS * (SESSIONS_PER_LEVEL - 1)    # 20 measured sessions per arm and level
PINNED_OK = {'registered': True, 'warning_absent': True}


def pinning_observation(log):
    """vLLM logs a successful cudaHostRegister only at DEBUG level; its size must be 17.18 GB."""
    size = re.escape(f'{HOST_BYTES / 1e9:.2f}')
    return {'registered': bool(re.search(r'cudaHostRegister rank=\d+ ' + size + ' GB', log)),
            'warning_absent': not any(s in log for s in ('cudaHostRegister failed', 'unpinned DMA'))}


def start_gate(arm, log, metrics, *, debug_pinning=False, pinning_start=None):
    """Check V for one server start, from its log and its first /metrics scrape."""
    if debug_pinning and arm not in NATIVE_ARMS:
        raise ValueError('A DEBUG start is only made for the two native arms')
    cfg = cache_config(metrics)
    blocks, size = int(cfg['num_gpu_blocks']), int(cfg['block_size'])
    pool = blocks * size
    failures = []
    logged = {int(x.replace(',', '')) for x in re.findall(r'GPU KV cache size: ([\d,]+) tokens', log)}
    if logged != {pool}:
        failures.append('Logged GPU KV capacity does not match cache configuration')
    if size != 16 or cfg.get('enable_prefix_caching') != 'True':
        failures.append('Block size 16 and prefix caching must be effective')
    pinned = pinning_observation(log)
    if arm in NATIVE_ARMS:
        if 'OffloadingConnector' not in log:
            failures.append('OffloadingConnector not logged')
        if debug_pinning:
            if not pinned['registered']:
                failures.append('DEBUG start shows no registration of the pinned host buffer')
        elif not (pinning_start and pinning_start.get('pass') is True
                  and pinning_start.get('arm') == arm
                  and pinning_start.get('debug_pinning') is True
                  and pinning_start.get('pinning') == PINNED_OK
                  and pinning_start.get('pinning_receipt')):
            failures.append("No verified DEBUG start of this arm in the start test")
        if not pinned['warning_absent']:
            failures.append('Host buffer registration failed')
    if arm == 'LAZY' and not re.search(r'SimpleCPUOffloadConnector:.*mode=lazy', log):
        failures.append('SimpleCPUOffloadConnector mode=lazy not logged')
    if arm in OFFLOAD_ARMS and not (max(LEVELS) + 1.5 * pool < 0.9 * (HOST_BYTES // BYTES_PER_TOKEN)):
        failures.append('Host buffer cannot hold the largest session plus the filler')
    if blocks <= 0:
        failures.append('No GPU KV blocks')
    return {'pass': not failures, 'failures': failures, 'arm': arm,
            'debug_pinning': debug_pinning, 'pinning': pinned, 'pool_tokens': pool,
            'pool_bytes': pool * BYTES_PER_TOKEN, 'cache_config': cfg}


def first_difference(ids, base):
    """0-based position of the first token where `ids` differs from `base`; None when they are equal."""
    for i, (a, b) in enumerate(zip(ids, base)):
        if a != b:
            return i
    return None if len(ids) == len(base) else min(len(ids), len(base))


def turn1_agreement(rounds):
    """Information only: per selected round and level, how many measured sessions generated the same
    turn-1 ids as the base arm, and how many had turn-1 prefix-cache hits.

    The base arm of a session is the arm of the first row with that session id (and a full
    512-id turn) in the round's sessions list, which is its execution order. This enters no
    check, no validity flag and no decision row.
    """
    out = {str(n): {} for n in LEVELS}
    for r in rounds:
        base = {}
        for s in r['sessions']:
            ids = s['turn1'].get('token_ids')
            if isinstance(ids, list) and len(ids) == GENERATED:
                base.setdefault(s['id'], s)
        for n in LEVELS:
            rows = [s for s in r['sessions'] if not s['warmup'] and s['N'] == n]
            compared = [s for s in rows if s['id'] in base and base[s['id']] is not s
                        and isinstance(s['turn1'].get('token_ids'), list)
                        and len(s['turn1']['token_ids']) == GENERATED]
            differing = [(s, first_difference(s['turn1']['token_ids'], base[s['id']]['turn1']['token_ids']))
                         for s in compared]
            differing = [{'arm': s['arm'], 'index': s['index'], 'base_arm': base[s['id']]['arm'],
                          'first_diff_from_base': d} for s, d in differing if d is not None]
            hits = [s for s in rows if isinstance((s['turn1'].get('delta') or {}).get('prefix_cache_hits'), (int, float))
                    and s['turn1']['delta']['prefix_cache_hits'] > 0]
            out[str(n)][str(r['round'])] = {
                'base_arms': sorted({base[s['id']]['arm'] for s in rows if s['id'] in base}),
                'agree': len(compared) - len(differing), 'compared': len(compared),
                'differing': differing,
                'turn1_prefix_hits_nonzero': len(hits), 'sessions': len(rows)}
    return out


def session_gates(record):
    """L, K, E, H and the token counts G, Hh, R of one session."""
    n, arm = record['N'], record['arm']
    first, resume = record['turn1'], record.get('resume')
    length = (isinstance(first.get('token_ids'), list) and len(first['token_ids']) == GENERATED
              and resume is not None and (resume.get('usage') or {}).get('prompt_tokens') == n)
    if not length:
        return {'L': False, 'E': None, 'K': None, 'H': None}
    d = resume['delta']
    if arm in NATIVE_ARMS:
        load = d.get('kv_offload_load_bytes')
        if (load is None and not record.get('warmup')) or (load is not None and (not math.isfinite(load) or load < 0)):
            raise ValueError('Required native kv_offload_load_bytes cross-check missing or invalid')
    g, h = d['prefix_cache_hits'], d['external_prefix_cache_hits']
    if any(not math.isfinite(x) or x < 0 or x != int(x) for x in (g, h)) or g + h > n:
        raise ValueError('Invalid token-counter accounting')
    if d['request_success'] != 1 or not (resume['ttft_ms'] > 0 and math.isfinite(resume['ttft_ms'])):
        raise ValueError('Request isolation or TTFT missing')
    load = d.get('kv_offload_load_bytes')
    return {'L': True, 'K': g >= reusable(n) - 16 if arm == 'KEEP' else None,
            'E': g <= 16 if arm != 'KEEP' else None,
            'H': h >= host_min(arm, n) if arm in OFFLOAD_ARMS else None,
            'G': g, 'Hh': h, 'R': n - g - h,
            'load_tokens_crosscheck': None if load is None else load / BYTES_PER_TOKEN,
            # Hh is counted at admission and not corrected if a load later fails; the bytes
            # loaded are whole blocks, so more than one block of difference is flagged.
            'load_mismatch': None if load is None else abs(load / BYTES_PER_TOKEN - h) > 16}


def quiet(samples, sockets):
    """Check Q over every telemetry segment (one per server start) of a round attempt."""
    reasons = []
    segments = samples if samples and isinstance(samples[0], list) else [samples]
    samples = [s for segment in segments for s in segment]
    if any(len(segment) < 2 for segment in segments):
        return {'pass': False, 'failures': ['Missing telemetry segment']}
    if len(samples) < 2 or any('error' in s for s in samples):
        return {'pass': False, 'failures': ['Telemetry missing or failed']}
    clocks = []
    foreign_ticks = {str(s): 0 for s in sockets}
    total_ticks = {str(s): 0 for s in sockets}
    for s in samples:
        if s['gpu']['foreign_pids']:
            reasons.append('Foreign GPU compute process')
        if not s['placement_ok']:
            reasons.append('Owned process/thread affinity escaped declared sockets')
        clocks.append(s['gpu']['clock_mhz'])
    for a, b in (pair for segment in segments for pair in zip(segment, segment[1:])):
        dt = b['time'] - a['time']
        if not 0 < dt <= 2.5:
            reasons.append('Telemetry interval gap')
        for socket, cpus in sockets.items():
            key = str(socket)
            try:
                total = sum(b['cpu'][str(c)]['total'] - a['cpu'][str(c)]['total'] for c in cpus)
                busy = sum(b['cpu'][str(c)]['busy'] - a['cpu'][str(c)]['busy'] for c in cpus)
                owned = b['owned_ticks'][key] - a['owned_ticks'][key]
                if total <= 0 or busy < 0 or owned < 0:
                    raise ValueError('Non-monotonic CPU accounting')
                total_ticks[key] += total
                foreign_ticks[key] += max(0, busy - owned)
            except (KeyError, ValueError):
                reasons.append('Incomplete CPU accounting')
    fractions = {s: foreign_ticks[s] / total_ticks[s] if total_ticks[s] else None for s in total_ticks}
    if any(v is None or v > .02 for v in fractions.values()):
        reasons.append('Foreign CPU time exceeds 2% on a socket')
    if any(not math.isfinite(c) or c <= 0 for c in clocks):
        reasons.append('Invalid GPU clock')
        median, low = None, None
    else:
        median = statistics.median(clocks)
        low = sum(c < .9 * median for c in clocks) / len(clocks)
        if low > .1:
            reasons.append('Low GPU clock in more than 10% of samples')
    return {'pass': not reasons, 'failures': sorted(set(reasons)), 'foreign_cpu_fraction': fractions,
            'median_clock_mhz': median, 'low_clock_fraction': low}


def compute(bundle):
    """Every check per arm and level, the per-round ratios, and the decision row.

    No missing round, failed arm or incomplete host copy can produce a winning row.
    """
    rounds = bundle['rounds']
    exhausted = any(r.get('quiet_exhausted') for r in rounds)
    selected = [r for r in rounds if r.get('selected')]
    if len({r['round'] for r in selected}) != len(selected):
        raise ValueError('Multiple selected attempts for one round')
    # The GPU pool depends on the compile-cache state of a start, not on the arm; a round whose
    # passing starts logged different pools is invalid for every arm.
    pool_mismatch = sorted(r['round'] for r in selected if len({
        s.get('pool_tokens') for s in r['starts'].values() if s.get('pass') is True}) > 1)
    records = [s for r in selected for s in r['sessions'] if not s['warmup']]
    cells = {}
    for arm in ARMS:
        for n in LEVELS:
            rows = [s for s in records if s['arm'] == arm and s['N'] == n]
            keys = [(s['round'], s['index']) for s in rows]
            if len(keys) != len(set(keys)):
                raise ValueError('Duplicate measurement')
            gs = [session_gates(s) for s in rows]
            ldrop = sum(not g['L'] for g in gs)
            edrop = sum(g['E'] is False for g in gs)
            # K describes the observed reuse, independently of KEEP's own drop count.
            keep_rows = [g for s, g in zip(rows, gs) if g['L'] and any(
                r['round'] == s['round'] and r['starts'].get(arm, {}).get('pass') is True
                for r in selected)]
            keep = arm != 'KEEP' or (bool(keep_rows) and all(g['K'] is True for g in keep_rows))
            host = arm in OFFLOAD_ARMS and bool(gs) and all(
                g['H'] is True for g in gs if g['L'] and g['E'] is not False)
            # Sessions dropped by L and by E share one "more than 1 of 20" count.
            valid = (len(rows) == MEASURED and len(selected) == ROUNDS and ldrop + edrop <= 1
                     and not pool_mismatch
                     and all(r['quiet']['pass'] for r in selected)
                     and all(r['starts'].get(arm, {}).get('pass') is True for r in selected))
            medians = {}
            for r in range(ROUNDS):
                vals = [s['resume']['ttft_ms'] for s, g in zip(rows, gs)
                        if s['round'] == r and g['L'] and g['E'] is not False and g['K'] is not False]
                if vals:
                    medians[str(r)] = statistics.median(vals)
            cells[f'{arm}/{n}'] = {'valid': valid and len(medians) == ROUNDS,
                'L_dropped': ldrop, 'E_dropped': edrop, 'K': keep,
                'load_mismatches': sum(g.get('load_mismatch') is True for g in gs if g['L']),
                'H': host,
                'host_status': ('complete' if host else 'host copy incomplete') if arm in OFFLOAD_ARMS
                               else 'not applicable',
                'median_ttft_ms': medians,
                'tokens': [{'round': s['round'], 'index': s['index'], **g} for s, g in zip(rows, gs)]}
    # Only K propagates across arms. No usable KEEP observation counts as a K failure.
    for n in LEVELS:
        if not cells[f'KEEP/{n}']['K']:
            for arm in ARMS:
                cells[f'{arm}/{n}']['valid'] = False
    ratios = {}
    for arm in OFFLOAD_ARMS:
        for n in LEVELS:
            c, drop = cells[f'{arm}/{n}'], cells[f'DROP/{n}']
            if c['valid'] and c['H'] and drop['valid']:
                ratios[f'{arm}/{n}'] = [c['median_ttft_ms'][str(r)] / drop['median_ttft_ms'][str(r)]
                                        for r in range(ROUNDS)]
    winners = [a for a in OFFLOAD_ARMS if all(
        f'{a}/{n}' in ratios and all(r <= RATIO_THRESHOLD for r in ratios[f'{a}/{n}']) for n in (8192, 28672))]
    # NO-WIN: every arm that passed H at 28672 must have r > 0.8 there in every round; an arm
    # that passed H but has no ratios cannot satisfy that.
    eligible = [a for a in OFFLOAD_ARMS if cells[f'{a}/28672']['H']]
    if exhausted or not cells['KEEP/28672']['K']:
        row = 'NOT-MEASURABLE'
    elif winners:
        row = 'RESTORE-WINS'
    elif eligible and all(f'{a}/28672' in ratios and all(r > RATIO_THRESHOLD for r in ratios[f'{a}/28672'])
                          for a in eligible):
        row = 'NO-WIN'
    else:
        row = 'INCONCLUSIVE'
    differences = {}
    for n in LEVELS:
        by_round = {}
        for r in range(ROUNDS):
            named = {a: cells[f'{a}/{n}'] for a in OFFLOAD_ARMS}
            item = {}
            for a, b, label in (('NATIVE', 'NATIVE-ALL', 'default_cost'),
                                ('NATIVE-ALL', 'LAZY', 'eager_minus_lazy')):
                if all(named[x]['valid'] and named[x]['H'] for x in (a, b)):
                    item[label + '_ttft_ms'] = named[a]['median_ttft_ms'][str(r)] - named[b]['median_ttft_ms'][str(r)]
                    rr = {x: [g['R'] for g in named[x]['tokens']
                              if g['round'] == r and g['L'] and g['E'] is not False] for x in (a, b)}
                    if all(rr.values()):
                        item[label + '_recomputed_tokens'] = statistics.median(rr[a]) - statistics.median(rr[b])
            by_round[str(r)] = item
        differences[str(n)] = by_round
    examine = []
    if row != 'RESTORE-WINS':
        examine.append(f'Row {row}: an outcome other than the expected one; examine it before explaining it')
    for key, c in cells.items():
        if not c['valid']:
            examine.append(f'{key}: not valid')
        if c['load_mismatches']:
            examine.append(f'{key}: {c["load_mismatches"]} sessions whose host-load bytes disagree with Hh')
        if key.split('/')[0] in OFFLOAD_ARMS and not c['H']:
            examine.append(f'{key}: host copy incomplete')
    return {'row': row, 'pool_mismatch_rounds': pool_mismatch, 'quiet_exhausted': exhausted,
            'selected_rounds': sorted(r['round'] for r in selected),
            'examine': examine, 'cells': cells, 'ratios_by_round': ratios,
            'differences_by_round': differences,
            'turn1_agreement_by_round': turn1_agreement(selected),
            'trace_pool_bytes_by_round': {str(r['round']): r['starts']['KEEP']['pool_bytes'] for r in selected
                                          if r['starts'].get('KEEP', {}).get('pass')}}
