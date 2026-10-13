"""Recompute every check, ratio and the decision row from a run directory's raw records."""
import json
import math
import statistics
from pathlib import Path

from .client import delta, first_token_ms, parse_metrics
from .gates import compute, quiet
from .spec import ARMS, LEVELS, OFFLOAD_ARMS, RATIO_THRESHOLD, REFERENCE_IDS_SHA256, ROUNDS, atomic_json


def _same(a, b):
    if a is None or b is None:
        return a is b
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)


def recompute_requests(bundle):
    """Counter deltas from the stored /metrics text, TTFT from the stored stream arrival times."""
    mismatches = []
    for r in bundle['rounds']:
        for s in r['sessions']:
            resume = s.get('resume')
            if not resume:
                continue
            where = f"round {r['round']} attempt {r['attempt']} {s['arm']} {s['id']}"
            if 'raw' in resume.get('before', {}) and 'raw' in resume.get('after', {}):
                d = delta(parse_metrics(resume['before']['raw']), parse_metrics(resume['after']['raw']))
                if not _same(d, resume['delta']):
                    mismatches.append(f'{where}: counter delta differs from the stored one')
                resume['delta'] = d
            if resume.get('events') and 'start_ns' in resume:
                ttft = first_token_ms(resume)
                if ttft is None or not _same(ttft, resume.get('ttft_ms')):
                    mismatches.append(f'{where}: TTFT differs from the stored one')
                resume['ttft_ms'] = ttft
    return mismatches


def recompute_quiet(bundle, run_dir, sockets):
    """Check Q from each attempt's telemetry files (one per server start that ran sessions)."""
    mismatches = []
    for r in bundle['rounds']:
        segments = []
        for group in r.get('group_order', []):
            path = run_dir / f"round-{r['round']}-attempt-{r['attempt']}-{group}" / 'telemetry.jsonl'
            if path.is_file():
                segments.append([json.loads(line) for line in path.read_text().splitlines() if line.strip()])
        if not segments:
            continue
        q = quiet(segments, sockets)
        if q['pass'] != r['quiet']['pass']:
            mismatches.append(f"round {r['round']} attempt {r['attempt']}: quiet check differs from the stored one")
        r['quiet'] = q
        r['selected'] = q['pass']
    return mismatches


def _fmt(x, digits=1):
    return '-' if x is None else f'{x:.{digits}f}'


def _median(values):
    return statistics.median(values) if values else None


def text(summary, bundle, manifest, notes):
    out = []
    w = out.append
    sessions = (manifest or {}).get('sessions', {})
    ids = sessions.get('ids_sha256')
    w(f"Sessions: ids_sha256 {ids or 'unknown'} (" + ('not recorded' if ids is None else
      'reference sessions' if ids == REFERENCE_IDS_SHA256 else 'NOT the reference sessions') + ')')
    model = (manifest or {}).get('model') or {}
    if 'reference' not in model:
        w('Model: unknown (not recorded)')
    elif model['reference']:
        w('Model: reference snapshot (config, generation config and tokenizer files match)')
    else:
        w('Model: NOT the reference snapshot (accepted with --allow-other-model); different files: '
          + ', '.join(model.get('different_files', [])))
    w(f"Rounds used: {summary['selected_rounds']} of {ROUNDS}; quiet reruns exhausted: {summary['quiet_exhausted']}")
    w('')
    w('Start check V and GPU KV pool (tokens) per round')
    w('round  ' + ''.join(f'{a:>16}' for a in ARMS))
    for r in bundle['rounds']:
        cells = []
        for a in ARMS:
            s = r['starts'].get(a)
            cells.append('-' if s is None else (f"{s.get('pool_tokens')}" if s.get('pass') else 'FAIL'))
        tag = '' if r.get('selected') else '  (not used)'
        w(f"{r['round']}.{r['attempt']}    " + ''.join(f'{c:>16}' for c in cells) + tag)
    if summary['pool_mismatch_rounds']:
        w(f"Rounds with different pools across starts (invalid for every arm): {summary['pool_mismatch_rounds']}")
    w('')
    w('Quiet check Q per round attempt: foreign CPU share per socket, share of low-clock samples')
    for r in bundle['rounds']:
        q = r.get('quiet', {})
        cpu = ', '.join(f'socket {k} {_fmt(100 * v, 2) if v is not None else "-"}%'
                        for k, v in (q.get('foreign_cpu_fraction') or {}).items())
        w(f"{r['round']}.{r['attempt']}  {'pass' if q.get('pass') else 'FAIL'}  {cpu}  "
          f"low clock {_fmt(100 * q['low_clock_fraction'], 1) if q.get('low_clock_fraction') is not None else '-'}%"
          + ('' if q.get('pass') else '  ' + '; '.join(q.get('failures', []))))
    w('')
    w('Per arm and level: checks, and the median over sessions of G, Hh, R (tokens)')
    w(f"{'cell':<18}{'valid':>6}{'Ldrop':>6}{'Edrop':>6}{'K':>6}{'host copy':>22}{'G':>8}{'Hh':>8}{'R':>8}{'load/57344':>12}{'mismatch':>9}")
    for a in ARMS:
        for n in LEVELS:
            c = summary['cells'][f'{a}/{n}']
            ok = [t for t in c['tokens'] if t['L']]
            loads = [t['load_tokens_crosscheck'] for t in ok if t.get('load_tokens_crosscheck') is not None]
            w(f"{a + '/' + str(n):<18}{str(c['valid']):>6}{c['L_dropped']:>6}{c['E_dropped']:>6}{str(c['K']):>6}"
              f"{c['host_status']:>22}{_fmt(_median([t['G'] for t in ok]), 0):>8}{_fmt(_median([t['Hh'] for t in ok]), 0):>8}"
              f"{_fmt(_median([t['R'] for t in ok]), 0):>8}{_fmt(_median(loads), 0):>12}{c['load_mismatches']:>9}")
    w('')
    w('Median TTFT per round (ms)')
    w(f"{'cell':<18}" + ''.join(f'{"round " + str(r):>11}' for r in range(ROUNDS)))
    for a in ARMS:
        for n in LEVELS:
            m = summary['cells'][f'{a}/{n}']['median_ttft_ms']
            w(f"{a + '/' + str(n):<18}" + ''.join(f'{_fmt(m.get(str(r))):>11}' for r in range(ROUNDS)))
    w('')
    w(f'r = median TTFT(arm) / median TTFT(DROP), per round (decision threshold {RATIO_THRESHOLD})')
    for a in OFFLOAD_ARMS:
        for n in LEVELS:
            key = f'{a}/{n}'
            ratios = summary['ratios_by_round'].get(key)
            w(f'{key:<18}' + ('no ratio (cell invalid or host copy incomplete)' if ratios is None
                              else ''.join(f'{x:>11.3f}' for x in ratios)))
    w('')
    w('Reported with every outcome: per level, the median over rounds of')
    for n in LEVELS:
        by_round = summary['differences_by_round'][str(n)]
        parts = []
        for label in ('default_cost_ttft_ms', 'default_cost_recomputed_tokens',
                      'eager_minus_lazy_ttft_ms', 'eager_minus_lazy_recomputed_tokens'):
            vals = [v[label] for v in by_round.values() if label in v]
            parts.append(f"{label} {_fmt(_median(vals))}")
        w(f'  N={n}: ' + ', '.join(parts))
    w('  (default cost = NATIVE minus NATIVE-ALL; eager minus lazy = NATIVE-ALL minus LAZY)')
    w('')
    w(f"Decision row: {summary['row']}")
    for item in summary['examine']:
        w(f'  examine: {item}')
    for note in notes:
        w(f'WARNING: {note}')
    return '\n'.join(out)


def main(args):
    run = Path(args.run)
    raw_path = run / 'raw.json' if run.is_dir() else run
    run_dir = raw_path.parent
    if not raw_path.is_file():
        print(f'error: {raw_path} not found')
        return 2
    bundle = json.loads(raw_path.read_text())
    manifest_path = run_dir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    notes = recompute_requests(bundle)
    if manifest and 'topology' in manifest:
        notes += recompute_quiet(bundle, run_dir, manifest['topology']['sockets'])
    else:
        notes.append('No manifest with the CPU topology: the quiet check is taken as stored')
    summary = compute(bundle)
    stored_path = run_dir / 'summary.json'
    if stored_path.is_file():
        stored = json.loads(stored_path.read_text())
        if stored.get('row') != summary['row']:
            notes.append(f"Stored decision row {stored.get('row')} differs from the recomputed {summary['row']}")
        paths = list(differences(stored, summary))
        if paths:
            shown = ', '.join(paths[:20]) + (f' and {len(paths) - 20} more' if len(paths) > 20 else '')
            notes.append(f'Stored summary.json differs from the recomputed one at {len(paths)} places: {shown}')
    print(text(summary, bundle, manifest, notes))
    if args.json:
        atomic_json(Path(args.json), {**summary, 'model': (manifest or {}).get('model'), 'warnings': notes})
    return 1 if notes and any(not n.startswith('No manifest') for n in notes) else 0


def differences(a, b, path=''):
    """Every place (as a key path) where two JSON values differ; numbers compared to 1e-9."""
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            sub = f'{path}.{key}' if path else str(key)
            if key not in a or key not in b:
                yield sub
            else:
                yield from differences(a[key], b[key], sub)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            yield path
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                yield from differences(x, y, f'{path}[{i}]')
    elif (isinstance(a, (int, float)) and isinstance(b, (int, float))
          and not isinstance(a, bool) and not isinstance(b, bool)):
        if not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9):
            yield path
    elif a != b or type(a) is not type(b):
        yield path
