"""Command line: python -m akvr {sessions,start-test,run,report,trace} ..."""
import argparse
import os
from pathlib import Path
import sys


def default_cache_dir():
    if os.environ.get('AKVR_CACHE_DIR'):
        return Path(os.environ['AKVR_CACHE_DIR'])
    base = Path(os.environ.get('XDG_CACHE_HOME') or Path.home() / '.cache')
    return base / 'agent-kv-retention'


def parser():
    p = argparse.ArgumentParser(prog='python -m akvr', description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)

    s = sub.add_parser('sessions', help='build the session prompts offline (CPU only)')
    s.add_argument('--model', type=Path, required=True,
                   help='local directory of Qwen/Qwen2.5-7B-Instruct (its tokenizer and config are read)')
    s.add_argument('--vllm-source', type=Path, required=True,
                   help='git checkout of vLLM at v0.30.0 (ced6857a); the prompt text is cut from it')
    s.add_argument('--output', type=Path, required=True, help='sessions file to write (must not exist)')
    s.add_argument('--allow-other-model', action='store_true',
                   help='accept a model directory whose files differ from the reference snapshot (recorded)')

    for name, text in (('start-test', 'start each configuration once, untimed (gate V)'),
                       ('run', 'the five timed rounds')):
        r = sub.add_parser(name, help=text)
        r.add_argument('--gpu', type=int, required=True, help='GPU index (nvidia-smi / PCI bus order)')
        r.add_argument('--model', type=Path, required=True, help='local model directory; served by this path')
        r.add_argument('--sessions', type=Path, required=True, help='file written by `sessions`')
        r.add_argument('--output', type=Path, required=True, help='new directory for this run')
        r.add_argument('--port', type=int, default=8066, help='server port (default 8066)')
        r.add_argument('--seed', type=int, default=603, help='seed of the arm order (default 603)')
        r.add_argument('--cache-dir', type=Path, default=default_cache_dir(),
                       help='compile caches of the servers; use the same one for start-test and run '
                            '(default $AKVR_CACHE_DIR, else $XDG_CACHE_HOME/agent-kv-retention, '
                            'else ~/.cache/agent-kv-retention)')
        r.add_argument('--lock-dir', type=Path, default=Path(os.environ.get('AKVR_LOCK_DIR', '/tmp')),
                       help='directory of the per-GPU lock file (default $AKVR_LOCK_DIR, else /tmp)')
        r.add_argument('--allow-other-model', action='store_true',
                       help='accept a model directory whose files differ from the reference snapshot '
                            '(recorded in manifest.json and printed by report)')
        r.add_argument('--lock', type=Path, action='append', default=[],
                       help='an extra lock file to hold exclusively for the whole run (repeatable)')
        if name == 'run':
            r.add_argument('--start-test', type=Path, required=True,
                           help='start-tests.json written by `start-test` with the same code and sessions')

    rep = sub.add_parser('report', help='recompute the checks and the decision row from a run')
    rep.add_argument('run', type=Path, help='run directory (or its raw.json)')
    rep.add_argument('--json', type=Path, help='also write the recomputed summary here')

    t = sub.add_parser('trace', help='count tool calls in the TraceLab trace (CPU only)')
    t.add_argument('--trace', type=Path, required=True,
                   help='syfi_coding_trace.jsonl.gz of TraceLab release v0.0.1 (sha256 checked)')
    t.add_argument('--output', type=Path, required=True, help='new directory for histogram.json and report.json')
    t.add_argument('--pools', type=Path, help="a run's summary.json; adds the shares that exceed pool / k")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == 'sessions':
        from .sessions import main as run
    elif args.command in ('start-test', 'run'):
        args.mode = 'start-test' if args.command == 'start-test' else 'rounds'
        from .run import main as run
    elif args.command == 'report':
        from .report import main as run
    else:
        from .trace import main as run
    return run(args) or 0


if __name__ == '__main__':
    sys.exit(main())
