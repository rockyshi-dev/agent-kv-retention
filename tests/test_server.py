"""Server ownership, environment, placement, locks, start check V and quiet check Q."""
import io
import json
import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import Mock, patch

from akvr import gates, server
from akvr.spec import COMMIT

from .helpers import fixture, telemetry_sample

METRICS = fixture('metrics_start.txt')


class Ownership(unittest.TestCase):
    def test_listener_inode_must_belong_to_server_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'net').mkdir()
            (root / '42/fd').mkdir(parents=True)
            (root / 'net/tcp').write_text('header\n0: 00000000:1F82 00000000:0000 0A 0:0 0:0 0 0 0 1234\n')
            (root / 'net/tcp6').write_text('header\n')
            (root / '42/fd/7').symlink_to('socket:[1234]')
            self.assertEqual(server.listener_ownership(8066, [42], root), ['1234'])
            with self.assertRaisesRegex(RuntimeError, 'port'):
                server.listener_ownership(8066, [99], root)

    def test_reparented_workers_remain_owned_but_reused_pids_do_not(self):
        table = {42: {'parent': 1, 'session': 10, 'start': 200, 'state': 'S'},
                 43: {'parent': 1, 'session': 43, 'start': 301, 'state': 'S'},
                 44: {'parent': 1, 'session': 44, 'start': 400, 'state': 'S'}}
        with patch.object(server, 'process_table', return_value=table):
            self.assertEqual(set(server.owned_tree(10, {43: 300, 44: 400})), {42, 44})

    def test_stop_waits_for_orphan_and_memory_after_parent_exit(self):
        child = {42: {'start': 200}}
        proc = Mock(pid=10, poll=Mock(return_value=0))
        busy = {'uuid': 'GPU-4', 'compute_pids': [], 'memory_mib': 1024}
        idle = busy | {'memory_mib': 0}
        with patch.object(server, 'owned_tree', side_effect=[child, {}, {}]), \
             patch.object(server, 'process_table', return_value=child), \
             patch.object(server, 'query_gpu', side_effect=[busy, busy, idle]) as query, \
             patch.object(server.os, 'kill') as kill, \
             patch.object(server.time, 'sleep') as sleep:
            server.stop_server(proc, {}, 4, 'GPU-4')
            kill.assert_called_once_with(42, signal.SIGTERM)
            self.assertEqual(sleep.call_count, 2)
            query.assert_called_with(4)

    def test_shutdown_timeout_never_waits_indefinitely(self):
        child = {42: {'start': 200}}
        proc = Mock(pid=10, poll=Mock(return_value=None))
        with patch.object(server, 'owned_tree', return_value=child), \
             patch.object(server, 'process_table', return_value=child), \
             patch.object(server, 'query_gpu', return_value={'uuid': 'GPU-4', 'compute_pids': [42], 'memory_mib': 1000}), \
             patch.object(server.time, 'monotonic', side_effect=[0, 0, 31, 60]), \
             patch.object(server.time, 'sleep'), patch.object(server.os, 'kill') as kill:
            with self.assertRaises(server.ShutdownError):
                server.stop_server(proc, {}, 4, 'GPU-4')
            self.assertEqual([c.args for c in kill.call_args_list], [(42, signal.SIGTERM), (42, signal.SIGKILL)])

    def test_gpu_placement_requires_owned_worker_only_on_its_gpu(self):
        with patch.object(server, 'processes', return_value={100: {}, 101: {}}), \
             patch.object(server.subprocess, 'check_output') as query:
            query.return_value = '101, GPU-4\n999, GPU-7\n'
            self.assertEqual(server.gpu_placement(100, 4, 'GPU-4')['compute_processes'],
                             [{'pid': 101, 'uuid': 'GPU-4'}])
            for raw in ('101, GPU-3\n', '101, GPU-4\n101, GPU-7\n', '999, GPU-4\n', '101, GPU-4\n999, GPU-4\n', ''):
                query.return_value = raw
                with self.subTest(raw=raw), self.assertRaises(RuntimeError):
                    server.gpu_placement(100, 4, 'GPU-4')

    def test_ready_checks_listener_model_and_gpu_before_recording_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'launch.json').write_text('{}')
            (root / 'server.log').write_text(fixture('server_native.log'))
            s = object.__new__(server.Server)
            s.directory, s.port, s.gpu, s.gpu_uuid, s.model = root, 8066, 4, 'GPU-4', Path('/model')
            s.process, s.known_processes = Mock(pid=10, poll=Mock(return_value=None)), {}

            def responses(model):
                health = io.BytesIO(b'')
                health.status = 200
                return [health, io.BytesIO(json.dumps({'data': [{'id': model}]}).encode())]
            with patch.object(server, 'owned_tree', return_value={10: {}, 42: {}}), \
                 patch.object(server, 'listener_ownership', return_value=['1234']) as listener, \
                 patch.object(server, 'gpu_placement', return_value={'uuid': 'GPU-4'}) as gpu, \
                 patch.object(server.urllib.request, 'urlopen', side_effect=responses('/model')) as url:
                s.ready()
                listener.assert_called_once_with(8066, {10: {}, 42: {}})
                gpu.assert_called_once_with(10, 4, 'GPU-4')
                record = json.loads((root / 'launch.json').read_text())
                self.assertEqual(record['listener_inodes'], ['1234'])
                self.assertEqual(record['default_sampling_params']['top_k'], 20)
                url.side_effect = responses('/someone-elses-model')
                with self.assertRaisesRegex(RuntimeError, 'model path'):
                    s.ready()


class Environment(unittest.TestCase):
    def test_runtime_override_guard_and_sampling_log_evidence(self):
        server.reject_overrides({'TORCH_HOME': '/shared/torch', 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'HF_HOME': '/x'})
        for key in ('LD_PRELOAD', 'PYTORCH_CUDA_ALLOC_CONF', 'TORCH_LOGS', 'TORCHINDUCTOR_CACHE_DIR',
                    'OMP_NUM_THREADS', 'VLLM_USE_V1', 'NCCL_DEBUG', 'CUDA_HOME', 'CUDA_VISIBLE_DEVICES',
                    'TRITON_INTERPRET', 'FLASHINFER_JIT_VERBOSE', 'NVIDIA_TF32_OVERRIDE',
                    'CUBLAS_WORKSPACE_CONFIG', 'LD_LIBRARY_PATH', 'PYTHONPATH'):
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, key):
                server.reject_overrides({key: 'inherited'})
        with self.assertRaisesRegex(RuntimeError, 'CUDA_DEVICE_ORDER'):
            server.reject_overrides({'CUDA_DEVICE_ORDER': 'FASTEST_FIRST'})
        obs = server.sampling_observation(fixture('server_lazy.log'))
        self.assertEqual(obs['default_sampling_params']['repetition_penalty'], 1.05)
        with self.assertRaises(RuntimeError):
            server.sampling_observation('no evidence')
        line = obs['default_sampling_params_log_lines'][0]
        with self.assertRaises(RuntimeError):
            server.sampling_observation(line + '\n' + line.replace('20', '10'))

    def test_environment_sets_device_order_offline_hub_and_caches(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch.dict(os.environ, {'HOME': tmp}, clear=True), \
             patch.object(server, 'check_versions'):
            env = server.environment(Path(tmp) / 'cache', 4)
            self.assertEqual((env['CUDA_VISIBLE_DEVICES'], env['CUDA_DEVICE_ORDER']), ('4', 'PCI_BUS_ID'))
            self.assertEqual(env['HF_HUB_OFFLINE'], '1')
            self.assertEqual(env['VLLM_CACHE_ROOT'], str(Path(tmp) / 'cache/vllm'))
            os.environ['VLLM_LOGGING_LEVEL'] = 'DEBUG'
            with self.assertRaisesRegex(RuntimeError, 'VLLM_LOGGING_LEVEL'):
                server.environment(Path(tmp) / 'cache', 4)

    def test_version_check_reads_installed_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'torch').mkdir()
            (root / 'vllm').mkdir()
            (root / 'torch/version.py').write_text('cuda: str = "12.9"\n')
            (root / 'vllm/_version.py').write_text(f"__version__ = '0.30.0+cu129'\ncommit = '{COMMIT[:9]}'\n")
            (root / 'bin').mkdir()
            (root / 'bin/vllm').write_text('')
            distribution = Mock(locate_file=lambda p: root / p)
            with patch.object(server.importlib.metadata, 'version', return_value='0.30.0+cu129'), \
                 patch.object(server.importlib.metadata, 'distribution', return_value=distribution), \
                 patch.object(server, 'vllm_binary', return_value=root / 'bin/vllm'):
                server.check_versions()
                (root / 'torch/version.py').write_text('cuda: str = "13.0"\n')
                with self.assertRaisesRegex(RuntimeError, 'CUDA 12.9'):
                    server.check_versions()
                (root / 'torch/version.py').write_text('cuda: str = "12.9"\n')
                (root / 'vllm/_version.py').write_text("commit = 'deadbeef0'\n")
                with self.assertRaisesRegex(RuntimeError, 'commit'):
                    server.check_versions()
            with patch.object(server.importlib.metadata, 'version', return_value='0.30.0'):
                with self.assertRaisesRegex(RuntimeError, 'vllm=='):
                    server.check_versions()

    def test_argv_differs_between_arms_only_by_the_connector(self):
        base = server.argv('KEEP', 8066, [8, 9], Path('/m'))
        self.assertEqual(base[:3], ['taskset', '-c', '8,9'])
        self.assertEqual(base[4:], ['serve', '/m', '--port', '8066'])
        lazy = server.argv('LAZY', 8066, [8, 9], Path('/m'))
        self.assertEqual(lazy[:len(base)], base)
        self.assertEqual(json.loads(lazy[-1]), {
            'kv_connector': 'SimpleCPUOffloadConnector', 'kv_role': 'kv_both',
            'kv_connector_extra_config': {'cpu_bytes_to_use': 17179869184, 'lazy_offload': True}})
        native_all = json.loads(server.argv('NATIVE-ALL', 8066, [8], Path('/m'))[-1])
        self.assertEqual(native_all['kv_connector_extra_config'],
                         {'cpu_bytes_to_use': 17179869184, 'offload_prompt_only': False})
        native = json.loads(server.argv('NATIVE', 8066, [8], Path('/m'))[-1])
        self.assertEqual(native['kv_connector'], 'OffloadingConnector')
        self.assertEqual(native['kv_connector_extra_config'], {'cpu_bytes_to_use': 17179869184})


class Redaction(unittest.TestCase):
    def test_secret_like_values_are_masked_case_insensitively(self):
        env = {'HF_TOKEN': 'hf_abc', 'openrouter_api_key': 'sk-1', 'AWS_SECRET_ACCESS_KEY': 'x',
               'DB_PASSWORD': 'p', 'GOOGLE_APPLICATION_CREDENTIALS': '/c.json', 'SSH_AUTH_SOCK': '/tmp/s',
               'ssh_connection': '1.2.3.4', 'HTTP_AUTH': 'basic', 'PATH': '/usr/bin', 'HOME': '/h',
               'CUDA_VISIBLE_DEVICES': '4'}
        out = server.redact(env)
        kept = {'PATH', 'HOME', 'CUDA_VISIBLE_DEVICES'}
        for key, value in env.items():
            self.assertEqual(out[key], value if key in kept else '<redacted>', key)
        self.assertEqual(env['HF_TOKEN'], 'hf_abc')  # the input is not changed


class Placement(unittest.TestCase):
    LSCPU = '# CPU,Core,Socket,Online\n' + ''.join(
        f'{cpu},{cpu % 8},{(cpu % 16) // 8},Y\n' for cpu in range(32))  # SMT siblings cpu and cpu+16

    def sysfs(self, root, node, cpulist):
        dev = root / 'bus/pci/devices/0000:81:00.0'
        dev.mkdir(parents=True)
        (dev / 'numa_node').write_text(f'{node}\n')
        n = root / f'devices/system/node/node{node}'
        n.mkdir(parents=True)
        (n / 'cpulist').write_text(cpulist + '\n')

    def test_server_on_gpu_socket_and_client_on_four_cores_of_the_other(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.sysfs(root, 1, '8-15,24-31')
            topo = server.topology(4, lscpu=self.LSCPU, bus='00000000:81:00.0', sysfs=root)
        self.assertEqual((topo['server_socket'], topo['client_socket']), ('1', '0'))
        self.assertEqual(topo['client'], [0, 16, 1, 17, 2, 18, 3, 19])
        self.assertEqual(topo['server'], list(range(8, 16)) + list(range(24, 32)))
        self.assertEqual(set(topo['sockets']), {'0', '1'})

    def test_gpu_on_socket_zero_moves_the_client(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.sysfs(root, 0, '0-7,16-23')
            topo = server.topology(0, lscpu=self.LSCPU, bus='00000000:81:00.0', sysfs=root)
        self.assertEqual((topo['server_socket'], topo['client_socket']), ('0', '1'))
        self.assertEqual(topo['client'], [8, 24, 9, 25, 10, 26, 11, 27])

    def test_one_socket_is_refused(self):
        lscpu = ''.join(f'{c},{c},0,Y\n' for c in range(8))
        with self.assertRaisesRegex(RuntimeError, 'Two CPU sockets'):
            server.topology(0, lscpu=lscpu, bus='00000000:81:00.0', sysfs=Path('/nonexistent'))

    def test_locks_exclude_another_owner(self):
        with tempfile.TemporaryDirectory() as d:
            paths = [Path(d) / 'gpu4.lock', Path(d) / 'run/owner.lock']
            first = server.Locks(paths)
            try:
                with self.assertRaises(BlockingIOError):
                    server.Locks([Path(d) / 'other.lock', paths[0]])
                self.assertEqual(paths[0].read_text().strip(), str(os.getpid()))
            finally:
                first.close()
            server.Locks(paths).close()


class StartCheck(unittest.TestCase):
    def test_debug_start_needs_registration_of_the_16_gib_buffer(self):
        debug = fixture('server_native_debug.log')
        start = gates.start_gate('NATIVE', debug, METRICS, debug_pinning=True)
        self.assertTrue(start['pass'], start['failures'])
        self.assertEqual(start['pool_tokens'], 118080)
        self.assertFalse(gates.start_gate('NATIVE', debug.replace('17.18 GB', '12.88 GB'), METRICS,
                                          debug_pinning=True)['pass'])
        self.assertFalse(gates.start_gate('NATIVE', fixture('server_native.log'), METRICS,
                                          debug_pinning=True)['pass'])
        with self.assertRaises(ValueError):
            gates.start_gate('LAZY', debug, METRICS, debug_pinning=True)

    def test_timed_native_start_needs_the_verified_debug_start_of_the_same_arm(self):
        timed = fixture('server_native.log')
        start = gates.start_gate('NATIVE', fixture('server_native_debug.log'), METRICS, debug_pinning=True)
        self.assertFalse(gates.start_gate('NATIVE', timed, METRICS)['pass'])
        start['pinning_receipt'] = {'log_path': 'checked by verify_pinning_start'}
        self.assertTrue(gates.start_gate('NATIVE', timed, METRICS, pinning_start=start)['pass'])
        self.assertFalse(gates.start_gate('NATIVE-ALL', timed, METRICS, pinning_start=start)['pass'])
        warned = timed + 'WARNING cudaHostRegister failed\n'
        self.assertFalse(gates.start_gate('NATIVE', warned, METRICS, pinning_start=start)['pass'])

    def test_lazy_mode_capacity_and_pool_match(self):
        lazy = fixture('server_lazy.log')
        self.assertTrue(gates.start_gate('LAZY', lazy, METRICS)['pass'])
        self.assertFalse(gates.start_gate('LAZY', lazy.replace('mode=lazy', 'mode=eager'), METRICS)['pass'])
        self.assertFalse(gates.start_gate('LAZY', lazy.replace('118,080', '84,864'), METRICS)['pass'])
        big = METRICS.replace('num_gpu_blocks="7380"', 'num_gpu_blocks="11000"')
        out = gates.start_gate('LAZY', lazy.replace('118,080', '176,000'), big)
        self.assertIn('Host buffer cannot hold the largest session plus the filler', out['failures'])
        # The default server has no host buffer, so no capacity condition applies.
        self.assertTrue(gates.start_gate('DEFAULT', lazy.replace('118,080', '176,000'), big)['pass'])
        no_cache = METRICS.replace('enable_prefix_caching="True"', 'enable_prefix_caching="False"')
        self.assertFalse(gates.start_gate('DEFAULT', lazy, no_cache)['pass'])


class QuietCheck(unittest.TestCase):
    SOCKETS = {'0': [0], '1': [1]}

    def test_quiet_accounts_for_both_sockets_and_missing_samples(self):
        a, b = telemetry_sample(0, 0, 0, 0), telemetry_sample(1, 1000, 100, 90)
        self.assertTrue(gates.quiet([a, b], self.SOCKETS)['pass'])
        b['owned_ticks']['1'] = 0
        self.assertFalse(gates.quiet([a, b], self.SOCKETS)['pass'])
        self.assertFalse(gates.quiet([a], self.SOCKETS)['pass'])
        self.assertFalse(gates.quiet([a, {'error': 'missing'}], self.SOCKETS)['pass'])

    def test_clock_gate_uses_whole_round_and_ignores_restart_gaps(self):
        segments = [[telemetry_sample(t + offset, t * 1000, 0, 0, 2000 if t == 0 and offset == 0 else 2520)
                     for t in range(10)] for offset in (0, 100, 200, 300)]
        self.assertTrue(gates.quiet(segments, self.SOCKETS)['pass'])
        for t in range(5):
            segments[1][t]['gpu']['clock_mhz'] = 2000
        self.assertIn('Low GPU clock in more than 10% of samples', gates.quiet(segments, self.SOCKETS)['failures'])
        segments[1] = [telemetry_sample(t + 100, t * 1000, 0, 0) for t in range(10)]
        segments[3][4]['gpu']['foreign_pids'] = [333]
        self.assertFalse(gates.quiet(segments, self.SOCKETS)['pass'])

    def test_interval_gap_inside_a_segment_fails(self):
        seg = [telemetry_sample(t, t * 1000, 0, 0) for t in (0, 1, 5)]
        self.assertIn('Telemetry interval gap', gates.quiet([seg], self.SOCKETS)['failures'])


if __name__ == '__main__':
    unittest.main()
