"""`report` and the command line, end to end on a synthetic run directory."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from akvr import __main__ as cli
from akvr.spec import GROUPS

from .helpers import bundle, telemetry_sample


REFERENCE_MODEL = {'reference': True, 'different_files': [], 'allow_other_model': False}


def write_run(root, data, foreign_cpu=False, model=REFERENCE_MODEL, ids=True):
    root.mkdir(parents=True, exist_ok=True)
    (root / 'raw.json').write_text(json.dumps(data))
    manifest = {'topology': {'sockets': {'0': [0], '1': [1]}}, 'model': model}
    if ids:
        manifest['sessions'] = {'ids_sha256': 'a766a2cbf6d492a4c285356f1bc745183c328aa4d3e0e45e80a1a5a8eb5dc5c9'}
    (root / 'manifest.json').write_text(json.dumps(manifest))
    for r in data['rounds']:
        for k, group in enumerate(GROUPS):
            d = root / f"round-{r['round']}-attempt-{r['attempt']}-{group}"
            d.mkdir()
            busy = 300 if foreign_cpu and r['round'] == 2 else 0
            rows = [telemetry_sample(100 * k + t, 1000 * t, busy * t, 0) for t in range(5)]
            (d / 'telemetry.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in rows))


def report(*argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = cli.main(['report', *map(str, argv)])
    return code, out.getvalue()


class Report(unittest.TestCase):
    def test_recomputes_row_and_quantities_from_raw_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / 'run'
            write_run(run, bundle(.7, raw=True))
            code, text = report(run, '--json', Path(tmp) / 'summary.json')
            self.assertEqual(code, 0, text)
            self.assertIn('Decision row: RESTORE-WINS', text)
            self.assertIn('(reference sessions)', text)
            self.assertNotIn('WARNING:', text)
            summary = json.loads((Path(tmp) / 'summary.json').read_text())
            self.assertEqual(summary['row'], 'RESTORE-WINS')
            self.assertEqual(summary['ratios_by_round']['NATIVE/28672'], [0.7] * 5)
            self.assertEqual(summary['warnings'], [])

    def test_ttft_is_taken_from_stream_arrival_times_not_the_stored_number(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = bundle(.7, raw=True)
            for r in data['rounds']:
                for s in r['sessions']:
                    if s['arm'] in ('NATIVE', 'NATIVE-ALL', 'LAZY'):
                        s['resume']['ttft_ms'] = 10.0  # stored value disagrees with the events (70 ms)
            write_run(Path(tmp) / 'run', data)
            (Path(tmp) / 'run/summary.json').write_text(json.dumps({'row': 'NO-WIN', 'ratios_by_round': {}}))
            code, text = report(Path(tmp) / 'run', '--json', Path(tmp) / 's.json')
            self.assertEqual(code, 1)
            self.assertIn('TTFT differs from the stored one', text)
            self.assertIn('Stored decision row NO-WIN differs', text)
            summary = json.loads((Path(tmp) / 's.json').read_text())
            self.assertAlmostEqual(summary['ratios_by_round']['LAZY/8192'][0], 0.7)

    def test_counters_are_taken_from_the_metrics_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = bundle(.7, raw=True)
            for r in data['rounds']:
                for s in r['sessions']:
                    if s['arm'] == 'KEEP':
                        s['resume']['delta']['prefix_cache_hits'] = 0  # the text says C hits
            write_run(Path(tmp) / 'run', data)
            code, text = report(Path(tmp) / 'run')
            self.assertEqual(code, 1)
            self.assertIn('counter delta differs', text)
            self.assertIn('Decision row: RESTORE-WINS', text)

    def test_quiet_check_is_recomputed_from_telemetry(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_run(Path(tmp) / 'run', bundle(.7, raw=True), foreign_cpu=True)
            code, text = report(Path(tmp) / 'run')
            self.assertEqual(code, 1)
            self.assertIn('round 2 attempt 0: quiet check differs', text)
            self.assertIn('Foreign CPU time exceeds 2% on a socket', text)
            self.assertIn('Decision row: INCONCLUSIVE', text)

    def test_model_and_sessions_status_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_run(Path(tmp) / 'a', bundle(.7, raw=True))
            self.assertIn('Model: reference snapshot', report(Path(tmp) / 'a')[1])
            other = {'reference': False, 'different_files': ['generation_config.json'], 'allow_other_model': True}
            write_run(Path(tmp) / 'b', bundle(.7, raw=True), model=other, ids=False)
            code, text = report(Path(tmp) / 'b', '--json', Path(tmp) / 'b.json')
            self.assertIn('Model: NOT the reference snapshot (accepted with --allow-other-model); '
                          'different files: generation_config.json', text)
            self.assertIn('ids_sha256 unknown (not recorded)', text)
            self.assertFalse(json.loads((Path(tmp) / 'b.json').read_text())['model']['reference'])

    def test_every_value_of_a_stored_summary_is_compared(self):
        from akvr.gates import compute
        with tempfile.TemporaryDirectory() as tmp:
            data = bundle(.7, raw=True)
            run = Path(tmp) / 'run'
            write_run(run, data)
            summary = compute(bundle(.7, raw=True))
            (run / 'summary.json').write_text(json.dumps(summary))
            code, text = report(run)
            self.assertEqual(code, 0, text)
            summary['cells']['KEEP/2048']['tokens'][3]['G'] = 1775
            summary['cells']['DROP/8192']['median_ttft_ms']['2'] = 101.0
            (run / 'summary.json').write_text(json.dumps(summary))
            code, text = report(run)
            self.assertEqual(code, 1)
            self.assertIn('differs from the recomputed one at 2 places', text)
            self.assertIn('cells.KEEP/2048.tokens[3].G', text)
            self.assertIn('cells.DROP/8192.median_ttft_ms.2', text)

    def test_missing_run_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, text = report(Path(tmp) / 'nothing')
        self.assertEqual(code, 2)


class CommandLine(unittest.TestCase):
    def test_help_and_required_arguments(self):
        for command in ([], ['sessions'], ['start-test'], ['run', '--gpu', '4'], ['trace']):
            with self.subTest(command=command), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit) as stop:
                cli.main(command)
            self.assertEqual(stop.exception.code, 2)
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit):
            cli.main(['run', '--help'])
        self.assertIn('--start-test', out.getvalue())

    def test_defaults(self):
        args = cli.parser().parse_args(['start-test', '--gpu', '4', '--model', '/m', '--sessions', 's.json',
                                        '--output', 'out'])
        self.assertEqual((args.port, args.seed, args.lock), (8066, 603, []))
        self.assertTrue(str(args.cache_dir).endswith('agent-kv-retention'))


if __name__ == '__main__':
    unittest.main()
