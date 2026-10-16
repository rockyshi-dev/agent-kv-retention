"""The session flow, arm-local failures, and the start test that the timed run depends on."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from akvr import run, server
from akvr.server import ShutdownError
from akvr.spec import ARMS, code_hashes, digest

from .helpers import fixture

METRICS = fixture('metrics_start.txt')


def args(mode, tmp, **kw):
    return SimpleNamespace(mode=mode, seed=603, port=8066, gpu=4, model=Path(tmp) / 'model',
                           cache_dir=Path(tmp) / 'cache', **kw)


class SessionFlow(unittest.TestCase):
    def material(self):
        return {'id': 'r0-n2048-s1', 'round': 0, 'N': 2048, 'index': 1, 'warmup': False,
                'text_sha256': 'x', 'prompt_ids': [1] * 1280, 'closing_ids': [2] * 256}

    def test_evicted_session_sends_turn1_fillers_then_the_exact_resume(self):
        client = Mock()
        generated = list(range(512))
        resume = {'usage': {'prompt_tokens': 2048}, 'ttft_ms': 50.0,
                  'delta': {'request_success': 1, 'prefix_cache_hits': 0, 'external_prefix_cache_hits': 1776,
                            'kv_offload_load_bytes': None}}
        client.request.side_effect = [{'token_ids': generated}, resume]
        filler = Mock()
        filler.evict.return_value = 180224
        with tempfile.TemporaryDirectory() as tmp, patch.object(run.time, 'sleep') as sleep:
            row = run.session(client, filler, self.material(), 'LAZY', 118080, Path(tmp), {})
            self.assertEqual(json.loads((Path(tmp) / 'sessions.jsonl').read_text())['id'], 'r0-n2048-s1')
        sleep.assert_called_once_with(2)
        filler.evict.assert_called_once()
        self.assertEqual(filler.evict.call_args.args[1], 118080)
        first, second = client.request.call_args_list
        self.assertEqual(first.args, ([1] * 1280, 512))
        self.assertEqual(first.kwargs, {'force': True})
        self.assertEqual(second.args[0], [1] * 1280 + generated + [2] * 256)
        self.assertEqual(second.args[1], 8)
        self.assertEqual(second.kwargs, {'stream': True})
        self.assertTrue(client.load_required)
        self.assertEqual((row['gates']['H'], row['gates']['R']), (True, 272))

    def test_keep_session_has_no_filler(self):
        client = Mock()
        client.request.side_effect = [{'token_ids': [3] * 512}, {
            'usage': {'prompt_tokens': 2048}, 'ttft_ms': 40.0,
            'delta': {'request_success': 1, 'prefix_cache_hits': 1776, 'external_prefix_cache_hits': 0}}]
        filler = Mock()
        with tempfile.TemporaryDirectory() as tmp, patch.object(run.time, 'sleep'):
            row = run.session(client, filler, self.material(), 'KEEP', 118080, Path(tmp), {})
        filler.evict.assert_not_called()
        self.assertTrue(row['gates']['K'])

    def test_first_arm_records_itself_as_base_with_its_ids(self):
        client = Mock()
        ids = list(range(512))
        client.request.side_effect = [{'token_ids': ids, 'delta': {'prefix_cache_hits': 0.0}}, {
            'usage': {'prompt_tokens': 2048}, 'ttft_ms': 90.0,
            'delta': {'request_success': 1, 'prefix_cache_hits': 0, 'external_prefix_cache_hits': 0}}]
        generated = {}
        with tempfile.TemporaryDirectory() as tmp, patch.object(run.time, 'sleep'):
            row = run.session(client, Mock(evict=Mock(return_value=1)), self.material(), 'DROP', 80000,
                              Path(tmp), generated)
            stored = json.loads((Path(tmp) / 'sessions.jsonl').read_text())
        self.assertEqual(generated['r0-n2048-s1'], {'arm': 'DROP', 'sha256': row['generated_sha256'], 'ids': ids})
        self.assertEqual((row['base_arm'], row['first_diff_from_base'], row['turn1_prefix_cache_hits']),
                         ('DROP', None, 0.0))
        self.assertEqual({k: stored[k] for k in ('base_arm', 'first_diff_from_base', 'generated_sha256')},
                         {k: row[k] for k in ('base_arm', 'first_diff_from_base', 'generated_sha256')})
        self.assertNotIn('error', stored)

    def test_generated_turn_that_differs_from_the_base_arm_is_recorded_and_the_session_continues(self):
        base_ids = [17] * 512
        own_ids = [17] * 40 + [18] + [17] * 471
        client = Mock()
        client.request.side_effect = [{'token_ids': own_ids, 'delta': {'prefix_cache_hits': 32.0}}, {
            'usage': {'prompt_tokens': 2048}, 'ttft_ms': 50.0,
            'delta': {'request_success': 1, 'prefix_cache_hits': 0, 'external_prefix_cache_hits': 1776,
                      'kv_offload_load_bytes': None}}]
        generated = {'r0-n2048-s1': {'arm': 'DROP', 'sha256': 'base', 'ids': base_ids}}
        with tempfile.TemporaryDirectory() as tmp, patch.object(run.time, 'sleep'):
            row = run.session(client, Mock(evict=Mock(return_value=1)), self.material(), 'LAZY', 80000,
                              Path(tmp), generated)
            stored = json.loads((Path(tmp) / 'sessions.jsonl').read_text())
        # No failure: the resume request is sent, built from this arm's own ids.
        self.assertEqual(client.request.call_count, 2)
        self.assertEqual(client.request.call_args_list[1].args[0], [1] * 1280 + own_ids + [2] * 256)
        self.assertEqual((row['base_arm'], row['first_diff_from_base'], row['turn1_prefix_cache_hits']),
                         ('DROP', 40, 32.0))
        self.assertNotEqual(row['generated_sha256'], 'base')
        self.assertTrue(row['gates']['L'])
        self.assertEqual(stored['first_diff_from_base'], 40)
        self.assertNotIn('error', stored)
        # The base arm stays the first one.
        self.assertEqual(generated['r0-n2048-s1']['arm'], 'DROP')

    def test_short_turn_one_has_no_base_and_fails_length(self):
        client = Mock()
        client.request.return_value = {'token_ids': [17] * 511, 'delta': {'prefix_cache_hits': 0.0}}
        with tempfile.TemporaryDirectory() as tmp:
            row = run.session(client, None, self.material(), 'LAZY', 80000, Path(tmp), {})
        self.assertFalse(row['gates']['L'])
        self.assertNotIn('base_arm', row)
        self.assertEqual(row['turn1_prefix_cache_hits'], 0.0)
        self.assertEqual(client.request.call_count, 1)


class Order(unittest.TestCase):
    def test_seed_603_gives_the_reference_order(self):
        """Server-start order per round, and the KEEP/DROP order on the shared default server."""
        import random
        groups, default = [], []
        for r in range(5):
            g = ['DEFAULT', 'NATIVE', 'NATIVE-ALL', 'LAZY']
            random.Random(603 + 100 * r).shuffle(g)
            a = ['KEEP', 'DROP']
            random.Random(603 + 100 * r).shuffle(a)
            groups.append(g)
            default.append(a)
        self.assertEqual(groups, [['DEFAULT', 'LAZY', 'NATIVE-ALL', 'NATIVE'],
                                  ['NATIVE-ALL', 'LAZY', 'NATIVE', 'DEFAULT'],
                                  ['NATIVE-ALL', 'NATIVE', 'LAZY', 'DEFAULT'],
                                  ['NATIVE', 'LAZY', 'NATIVE-ALL', 'DEFAULT'],
                                  ['DEFAULT', 'NATIVE', 'NATIVE-ALL', 'LAZY']])
        self.assertEqual(default, [['DROP', 'KEEP']] * 4 + [['KEEP', 'DROP']])


class ArmFailures(unittest.TestCase):
    def run_groups(self, root, groups, gates, mode='start-test', materials=(), session=None, shutdown_error=None):
        servers = []

        def make(group, a, topo, directory, locks, **kwargs):
            directory.mkdir()
            (directory / 'server.log').write_text('fixture')
            obj = Mock(process=Mock(pid=123))
            obj.completed.side_effect = shutdown_error
            servers.append(obj)
            return obj
        c = Mock(last_record=None, last_metrics_raw=None)
        c.metrics.return_value = {'raw': 'fixture'}
        with patch.object(run, 'Server', side_effect=make), patch.object(run, 'Client', return_value=c), \
             patch.object(run, 'start_gate', side_effect=gates), patch.object(run, 'Recorder') as recorder, \
             patch.object(run, 'session', side_effect=session):
            recorder.return_value.close.return_value = []
            out = [run.execute_group(group, 0, 0, args(mode, root), {'sockets': {}}, None,
                                     materials, None, {}, root / group) for group in groups]
        return out, servers

    def test_failed_start_check_is_recorded_stopped_and_other_group_continues(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = {'pass': True, 'pool_tokens': 80000}
            bad = {'pass': False, 'failures': ['missing lazy mode']}
            result, servers = self.run_groups(root, ['LAZY', 'DEFAULT'], [bad, good, good])
            self.assertFalse(result[0]['starts']['LAZY']['pass'])
            self.assertEqual(result[0]['sessions'], [])
            self.assertTrue(result[1]['starts']['KEEP']['pass'])
            self.assertTrue(result[1]['starts']['DROP']['pass'])
            self.assertIn('missing lazy mode', (root / 'LAZY/failure.json').read_text())
            for s in servers:
                s.completed.assert_called()

    def test_arm_failure_drops_only_that_arm_for_the_round(self):
        calls = []

        def session(client, filler, material, arm, pool, directory, generated):
            calls.append(arm)
            if arm == 'DROP':
                raise run.ArmFailure('Warm-up failed the length check')
            return {'arm': arm, 'gates': {'L': True}}
        materials = [{'round': 0, 'N': 2048, 'index': i, 'warmup': i == 0} for i in range(5)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = {'pass': True, 'pool_tokens': 80000}
            out, servers = self.run_groups(root, ['DEFAULT', 'LAZY'], [good] * 4, 'rounds', materials, session)
            self.assertTrue(out[0]['starts']['KEEP']['pass'])
            self.assertFalse(out[0]['starts']['DROP']['pass'])
            self.assertEqual(len(out[0]['sessions']), 5)
            self.assertEqual(calls.count('DROP'), 1)
            self.assertTrue(out[1]['starts']['LAZY']['pass'])
            self.assertIn('Warm-up failed', (root / 'DEFAULT/failure.json').read_text())

    def test_engine_crash_mid_round_keeps_other_arms(self):
        def session(client, filler, material, arm, pool, directory, generated):
            if arm == 'LAZY' and material['index'] == 0:
                raise RuntimeError('Completion HTTP 500: EngineDeadError')
            return {'arm': arm, 'gates': {'L': True}}
        materials = [{'round': 0, 'N': n, 'index': i, 'warmup': i == 0} for n in (2048, 8192, 28672) for i in range(5)]
        with tempfile.TemporaryDirectory() as tmp:
            out, servers = self.run_groups(Path(tmp), ['LAZY', 'NATIVE'], [{'pass': True, 'pool_tokens': 1}] * 4,
                                           'rounds', materials, session)
        self.assertFalse(out[0]['starts']['LAZY']['pass'])
        self.assertEqual(out[0]['sessions'], [])
        self.assertTrue(out[1]['starts']['NATIVE']['pass'])
        self.assertEqual(len(out[1]['sessions']), 15)

    def test_failed_warmup_is_arm_failure_not_an_ignored_trial(self):
        material = [{'round': 0, 'N': 2048, 'index': 0, 'warmup': True}]
        with tempfile.TemporaryDirectory() as tmp:
            out, _ = self.run_groups(Path(tmp), ['LAZY'], [{'pass': True, 'pool_tokens': 80000}] * 2,
                                     'rounds', material, lambda *a: {'arm': 'LAZY', 'gates': {'L': False}})
            self.assertFalse(out[0]['starts']['LAZY']['pass'])
            self.assertEqual(out[0]['sessions'], [])

    def test_shutdown_failure_prevents_next_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ShutdownError):
                self.run_groups(root, ['LAZY', 'DEFAULT'], [{'pass': False, 'failures': ['V failed']}],
                                shutdown_error=ShutdownError('still live after 60 seconds'))
            self.assertFalse((root / 'DEFAULT').exists())


class StartTestReceipts(unittest.TestCase):
    def test_debug_start_then_default_timed_start_through_the_real_launcher(self):
        captured = []

        def fake_process(command, **kwargs):
            captured.append(kwargs['env'].copy())
            log = fixture('server_native.log')
            if kwargs['env'].get('VLLM_LOGGING_LEVEL') == 'DEBUG':
                log = fixture('server_native_debug.log')
            kwargs['stdout'].write(log)
            kwargs['stdout'].flush()
            return Mock(pid=123, poll=Mock(return_value=0))

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(server, 'environment', side_effect=lambda *a: {'PATH': '/bin', 'HF_TOKEN': 'hf_secret', 'SSH_AUTH_SOCK': '/s'}), \
             patch.object(server, 'query_gpu', return_value={'uuid': 'GPU-4', 'compute_pids': [], 'memory_mib': 0}), \
             patch.object(server.socket, 'socket') as sockets, \
             patch.object(server.subprocess, 'Popen', side_effect=fake_process), \
             patch.object(server.importlib.metadata, 'distributions', return_value=[]), \
             patch.object(server.Server, 'ready'), \
             patch.object(server, 'owned_tree', return_value={}), \
             patch.object(server, 'stop_server', return_value={'live_pids': []}), \
             patch.object(run, 'Client') as client_class, patch.object(run, 'Recorder'):
            client_class.return_value.metrics.return_value = {'raw': METRICS}
            (Path(tmp) / 'model').mkdir()
            (Path(tmp) / 'model/generation_config.json').write_text('{"top_k": 20}')
            topo = {'server': [4], 'sockets': {'0': [0], '1': [4]}}
            starts = {}
            for arm in ('NATIVE', 'NATIVE-ALL'):
                for mode in ('start-test', 'rounds'):
                    out = run.execute_group(arm, 0, 0, args(mode, tmp), topo, Mock(fds=()), [], None, {},
                                            Path(tmp) / (arm + '-' + mode), pinning_start=starts.get(arm))
                    self.assertTrue(out['starts'][arm]['pass'], out)
                    if mode == 'start-test':
                        starts[arm] = out['starts'][arm]
                        server.verify_pinning_start(arm, starts[arm])
                    else:
                        self.assertFalse(out['starts'][arm]['debug_pinning'])
            self.assertEqual([e.get('VLLM_LOGGING_LEVEL') for e in captured], ['DEBUG', None, 'DEBUG', None])
            # The server gets the real environment; the record has secrets masked.
            self.assertEqual(captured[0]['HF_TOKEN'], 'hf_secret')
            recorded = json.loads((Path(tmp) / 'NATIVE-start-test/launch.json').read_text())['environment']
            self.assertEqual((recorded['HF_TOKEN'], recorded['SSH_AUTH_SOCK'], recorded['PATH']),
                             ('<redacted>', '<redacted>', '/bin'))
            self.assertNotIn('hf_secret', (Path(tmp) / 'NATIVE-start-test/launch.json').read_text())
            probe = sockets.return_value.__enter__.return_value
            probe.setsockopt.assert_called_with(server.socket.SOL_SOCKET, server.socket.SO_REUSEADDR, 1)

            material = Path(tmp) / 'sessions.json'
            material.write_text('fixture')
            start_path = Path(tmp) / 'start-tests.json'
            artifact = {'mode': 'start-test', 'code_sha256': code_hashes(), 'sessions_sha256': digest(material),
                        'starts': {a: starts.get(a, {'pass': True}) for a in ARMS}}
            start_path.write_text(json.dumps(artifact))
            run_args = args('rounds', tmp)
            (Path(tmp) / 'manifest.json').write_text(json.dumps(
                {'code_sha256': code_hashes(), 'identity': server.run_identity(run_args)}))
            self.assertEqual(server.verify_start_tests(start_path, material, run_args)['starts'], artifact['starts'])
            # The timed run must use the start test's GPU, model directory, cache directory and interpreter.
            for key, value in (('gpu', 5), ('model', Path(tmp) / 'other-model'), ('cache_dir', Path(tmp) / 'cold')):
                with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, 'different ' + key):
                    server.verify_start_tests(start_path, material, SimpleNamespace(**{**vars(run_args), key: value}))
            with patch.object(server.sys, 'executable', '/other/python'), \
                    self.assertRaisesRegex(RuntimeError, 'different python'):
                server.verify_start_tests(start_path, material, run_args)
            (Path(tmp) / 'manifest.json').unlink()
            with self.assertRaisesRegex(RuntimeError, 'manifest.json'):
                server.verify_start_tests(start_path, material, run_args)
            (Path(tmp) / 'manifest.json').write_text(json.dumps(
                {'code_sha256': code_hashes(), 'identity': server.run_identity(run_args)}))
            # Another code version or another sessions file cannot release the timed run.
            for key, value in (('code_sha256', {}), ('sessions_sha256', 'other'), ('mode', 'rounds')):
                changed = copy.deepcopy(artifact)
                changed[key] = value
                start_path.write_text(json.dumps(changed))
                with self.subTest(key=key), self.assertRaises(RuntimeError):
                    server.verify_start_tests(start_path, material, run_args)
            missing = copy.deepcopy(artifact)
            del missing['starts']['LAZY']
            start_path.write_text(json.dumps(missing))
            with self.assertRaises(RuntimeError):
                server.verify_start_tests(start_path, material, run_args)
            start_path.write_text(json.dumps(artifact))
            # A warning appended to the DEBUG log after the fact is caught by its hash and its text.
            log = Path(starts['NATIVE']['pinning_receipt']['log_path'])
            log.write_text(log.read_text() + 'cudaHostRegister failed\n')
            with self.assertRaises(RuntimeError):
                server.verify_start_tests(start_path, material, run_args)
            starts['NATIVE']['pinning_receipt']['log_sha256'] = digest(log)
            with self.assertRaises(RuntimeError):
                server.verify_pinning_start('NATIVE', starts['NATIVE'])


if __name__ == '__main__':
    unittest.main()
