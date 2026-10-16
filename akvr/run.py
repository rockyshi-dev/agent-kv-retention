"""The start test and the timed rounds."""
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

from .client import Client, Filler
from .gates import compute, first_difference, quiet, session_gates, start_gate
from .server import Locks, Server, ShutdownError, run_identity, topology, verify_start_tests
from .sessions import validate
from .spec import (CLOSING, GENERATED, GROUPS, LEVELS, QUIET_ATTEMPTS, RESUME_OUTPUT, RESUME_WAIT_S, ROUNDS,
                   atomic_json, check_model, code_hashes, digest, ids_digest)
from .telemetry import Recorder

NATIVE_ARMS = ('NATIVE', 'NATIVE-ALL')


def append(path, value):
    with path.open('a') as f:
        f.write(json.dumps(value, separators=(',', ':')) + '\n')
        f.flush()


class ArmFailure(RuntimeError):
    """This arm cannot be measured in this round; its raw error is kept."""


def session(client, filler, material, arm, pool, directory, generated):
    """Turn 1, eviction (all arms but KEEP), a 2 s wait, then the timed resume request.

    The generated turn is fixed in length (512 ids), not in content: each arm's own server
    generates it, and the resume request is built from that arm's ids. For information only,
    `generated` keeps the first arm's ids per session (the base arm), so that every later row
    records whether, and from which position, its ids differ from the base arm's.
    """
    row = {k: material[k] for k in ('id', 'round', 'N', 'index', 'warmup', 'text_sha256')}
    row['arm'] = arm
    row['turn1'] = client.request(material['prompt_ids'], GENERATED, force=True)
    row['turn1_prefix_cache_hits'] = (row['turn1'].get('delta') or {}).get('prefix_cache_hits')
    ids = row['turn1'].get('token_ids')
    if not isinstance(ids, list) or len(ids) != GENERATED:
        row['gates'] = session_gates(row)
        append(directory / 'sessions.jsonl', row)
        return row
    identity = hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
    base = generated.setdefault(material['id'], {'arm': arm, 'sha256': identity, 'ids': list(ids)})
    row['generated_sha256'] = identity
    row['base_arm'] = base['arm']
    row['first_diff_from_base'] = first_difference(ids, base['ids'])
    row['filler_tokens'] = 0
    if arm != 'KEEP':
        row['filler_tokens'] = filler.evict(client, pool,
            lambda r: append(directory / 'filler.jsonl', {'session': material['id'], 'arm': arm, 'request': r}))
    row['wait_started_ns'] = time.monotonic_ns()
    time.sleep(RESUME_WAIT_S)
    row['wait_ended_ns'] = time.monotonic_ns()
    resume_ids = material['prompt_ids'] + ids + material['closing_ids']
    if len(resume_ids) != material['N'] or len(material['closing_ids']) != CLOSING:
        raise ValueError('Resume token length differs from N')
    row['resume'] = client.request(resume_ids, RESUME_OUTPUT, stream=True)
    client.load_required = True
    row['gates'] = session_gates(row)
    append(directory / 'sessions.jsonl', row)
    return row


def execute_group(group, round_id, attempt, args, topo, locks, materials, filler, generated, directory,
                  pinning_start=None):
    """One server start: KEEP and DROP share the default server, each offload arm has its own."""
    server = None
    client = None
    recorder = None
    startup = True
    debug_pinning = args.mode == 'start-test' and group in NATIVE_ARMS
    arms = ['KEEP', 'DROP'] if group == 'DEFAULT' else [group]
    random.Random(args.seed + round_id * 100 + attempt).shuffle(arms)
    result = {'starts': {}, 'sessions': [], 'observations': None, 'errors': [], 'arm_order': list(arms)}

    def fail(affected, error):
        result['errors'].append({'arms': affected, 'error': repr(error)})
        for arm in affected:
            result['starts'][arm] = {**result['starts'].get(arm, {}),
                                    'pass': False, 'failures': [repr(error)]}
        # Earlier samples of this arm stay in sessions.jsonl as raw evidence.
        result['sessions'] = [s for s in result['sessions'] if s['arm'] not in affected]
        atomic_json(directory / 'failure.json', {'errors': result['errors'], 'partial': result,
            'last_request': client.last_record if client else None,
            'last_metrics_raw': client.last_metrics_raw if client else None})
    try:
        server = Server(group, args, topo, directory, locks, debug_pinning=debug_pinning)
        server.ready()
        client = Client(args.port, str(args.model), arm=group)
        metrics = client.metrics()
        log = (directory / 'server.log').read_text()
        gate = start_gate(group, log, metrics['raw'], debug_pinning=debug_pinning, pinning_start=pinning_start)
        atomic_json(directory / 'start.json', {'gate': gate, 'metrics': metrics})
        result['starts'] = {arm: dict(gate) for arm in arms}
        if not gate['pass']:
            raise ArmFailure('Start check V failed; arm not measured: ' + '; '.join(gate['failures']))
        startup = False
        if args.mode == 'rounds':
            recorder = Recorder(directory / 'telemetry.jsonl', topo, server.process.pid)
            recorder.start()
            for arm in arms:
                try:
                    for n in LEVELS:
                        rows = sorted([s for s in materials if s['round'] == round_id and s['N'] == n],
                                      key=lambda s: s['index'])
                        for item in rows:
                            record = session(client, filler, item, arm, gate['pool_tokens'], directory, generated)
                            if item['warmup'] and not record['gates']['L']:
                                raise ArmFailure('Warm-up failed the length check')
                            result['sessions'].append(record)
                except Exception as error:
                    fail([arm], error)
            observations = recorder.close()
            recorder = None
            result['observations'] = observations
        server.completed()
        # Re-read the complete log, including shutdown, before sealing the receipt.
        gate = start_gate(group, (directory / 'server.log').read_text(), metrics['raw'],
                          debug_pinning=debug_pinning, pinning_start=pinning_start)
        if debug_pinning and gate['pass']:
            gate['pinning_receipt'] = {key: value for kind, filename in
                (('log', 'server.log'), ('launch', 'launch.json')) for key, value in
                ((kind + '_path', str((directory / filename).resolve())),
                 (kind + '_sha256', digest(directory / filename)))}
        elif pinning_start:
            gate['pinning_receipt'] = pinning_start['pinning_receipt']
        for arm in arms:
            if result['starts'][arm]['pass']:
                result['starts'][arm] = dict(gate)
        atomic_json(directory / 'start.json', {'gate': gate, 'metrics': metrics})
        if not gate['pass']:
            fail(arms, ArmFailure('Start check V failed in the complete server log: ' + '; '.join(gate['failures'])))
        return result
    except BaseException as error:
        fail(arms, error)
        if isinstance(error, ShutdownError):
            raise
        if startup and isinstance(error, Exception):
            return result  # `finally` proves shutdown before another arm can start
        raise
    finally:
        try:
            if recorder is not None:
                recorder.close()
        finally:
            try:
                if client is not None:
                    client.close()
            finally:
                if server is not None and not isinstance(sys.exc_info()[1], ShutdownError):
                    server.completed()


def main(args):
    sessions_path = Path(args.sessions)
    data = json.loads(sessions_path.read_text())
    validate(data)
    model = check_model(args.model, args.allow_other_model)
    built = data.get('tokenizer_files') or {}
    clash = sorted(k for k, v in built.items() if model['files_sha256'].get(k, v) != v)
    if clash and not args.allow_other_model:
        raise ValueError('Sessions were built with other model files: ' + ', '.join(clash))
    prereq = verify_start_tests(args.start_test, sessions_path, args) if args.mode == 'rounds' else None
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    gpu_lock = Path(args.lock_dir) / f'agent-kv-retention-gpu{args.gpu}.lock'
    locks = Locks([gpu_lock, *args.lock, output / 'owner.lock'])
    try:
        topo = topology(args.gpu)
        os.sched_setaffinity(0, topo['client'])
        atomic_json(output / 'manifest.json', {'mode': args.mode, 'argv': sys.argv,
            'topology': topo, 'model': model, 'identity': run_identity(args), 'code_sha256': code_hashes(),
            'sessions': {'path': str(sessions_path.resolve()), 'sha256': digest(sessions_path),
                         'ids_sha256': ids_digest(data['sessions'])},
            'start_test': None if prereq is None else {k: prereq[k] for k in ('path', 'sha256')},
            'seed': args.seed, 'port': args.port, 'gpu': args.gpu, 'cache_dir': str(args.cache_dir),
            'locks': [str(gpu_lock), *map(str, args.lock)], 'started_wall_time': time.time()})
        filler = Filler(data['filler_vocabulary'], [s['prompt_ids'][:16] for s in data['sessions']])
        starts = prereq['starts'] if prereq else None
        bundle = {'rounds': [], 'manifest': str(output / 'manifest.json')}
        for round_id in range(1 if args.mode == 'start-test' else ROUNDS):
            for attempt in range(1 if args.mode == 'start-test' else QUIET_ATTEMPTS):
                row = {'round': round_id, 'attempt': attempt, 'starts': {}, 'sessions': []}
                generated = {}  # turn-1 ids of the first arm per session, this round attempt
                groups = list(GROUPS)
                random.Random(args.seed + round_id * 100 + attempt).shuffle(groups)
                row['group_order'] = groups
                observations = []
                for group in groups:
                    arms = ['KEEP', 'DROP'] if group == 'DEFAULT' else [group]
                    if starts is not None and any(not starts[a]['pass'] for a in arms):
                        row['starts'].update({a: starts[a] for a in arms})
                        continue
                    directory = output / f'round-{round_id}-attempt-{attempt}-{group}'
                    out = execute_group(group, round_id, attempt, args, topo, locks,
                        data['sessions'], filler, generated, directory,
                        pinning_start=starts[group] if starts and group in NATIVE_ARMS else None)
                    row['starts'].update(out['starts'])
                    row.setdefault('arm_order', {})[group] = out['arm_order']
                    row['sessions'].extend(out['sessions'])
                    if out['observations'] is not None:
                        observations.append(out['observations'])
                if args.mode == 'start-test':
                    row.update(mode=args.mode, code_sha256=code_hashes(), sessions_sha256=digest(sessions_path))
                    atomic_json(output / 'start-tests.json', row)
                    print(json.dumps({a: {'pass': s['pass'], 'pool_tokens': s.get('pool_tokens'),
                                          'failures': s.get('failures', [])} for a, s in row['starts'].items()},
                                     indent=2))
                    return 0 if all(s['pass'] for s in row['starts'].values()) else 1
                row['quiet'] = quiet(observations, topo['sockets'])
                row['selected'] = row['quiet']['pass']
                row['quiet_exhausted'] = not row['selected'] and attempt == QUIET_ATTEMPTS - 1
                bundle['rounds'].append(row)
                atomic_json(output / 'raw.json', bundle)
                if row['selected'] or row['quiet_exhausted']:
                    break
            if row['quiet_exhausted']:
                break
        summary = compute(bundle)
        atomic_json(output / 'summary.json', summary)
        print(json.dumps({'row': summary['row'], 'examine': summary['examine']}, indent=2))
        return 0
    finally:
        locks.close()
