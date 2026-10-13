"""Environment checks, CPU placement, locks and the lifecycle of one vLLM server."""
import ast
import csv
import fcntl
import importlib.metadata
import io
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request

from .spec import (ARMS, COMMIT, TORCH_CUDA, VLLM_VERSION, atomic_json, code_hashes, config, digest)
from .telemetry import processes, query_gpu

NATIVE_ARMS = ('NATIVE', 'NATIVE-ALL')


class ShutdownError(RuntimeError):
    """Cleanup did not prove that the server and its GPU memory are gone; nothing else may start."""


# ---------------------------------------------------------------- process ownership

def process_table(proc=Path('/proc')):
    table = {}
    for path in proc.iterdir():
        if not path.name.isdigit():
            continue
        try:
            f = (path / 'stat').read_text().rsplit(')', 1)[1].split()
            table[int(path.name)] = {'parent': int(f[1]), 'session': int(f[3]),
                                     'start': int(f[19]), 'state': f[0]}
        except (FileNotFoundError, ProcessLookupError):
            continue
    return table


def owned_tree(root, known):
    """The server's session and descendants, including workers reparented away from it."""
    table = process_table()
    owned = {p for p, rec in table.items() if
             (p in known and known[p] == rec['start']) or rec['session'] == root}
    while True:
        children = {p for p, rec in table.items() if rec['parent'] in owned}
        if children <= owned:
            break
        owned |= children
    known.update({p: table[p]['start'] for p in owned})
    return {p: table[p] for p in owned if table[p]['state'] not in ('Z', 'X')}


def listener_ownership(port, pids, proc=Path('/proc')):
    """The listening socket on `port` must belong to the server's processes only."""
    inodes = set()
    for name, addresses in (
        ('tcp', {'00000000', '0100007F'}),
        ('tcp6', {'0' * 32, '0000000000000000FFFF00000100007F'}),
    ):
        for line in (proc / 'net' / name).read_text().splitlines()[1:]:
            fields = line.split()
            address, encoded_port = fields[1].split(':')
            if fields[3] == '0A' and address in addresses and int(encoded_port, 16) == port:
                inodes.add(fields[9])
    owned = set()
    for pid in pids:
        try:
            for fd in (proc / str(pid) / 'fd').iterdir():
                try:
                    target = os.readlink(fd)
                except (FileNotFoundError, ProcessLookupError):
                    continue
                if target.startswith('socket:['):
                    owned.add(target[8:-1])
        except (FileNotFoundError, ProcessLookupError):
            continue
    if not inodes or not inodes <= owned:
        raise RuntimeError('Answering port is not owned exclusively by this server tree')
    return sorted(inodes)


def stop_server(process, known, gpu, expected_uuid, timeout=60):
    """TERM, then KILL after half the timeout; done only when the tree is gone and the GPU is free."""
    started = time.monotonic()
    term_sent, kill_sent = set(), set()
    last = {}
    while True:
        tree = owned_tree(process.pid, known)
        process.poll()  # reap the direct child; descendants may outlive it
        elapsed = time.monotonic() - started
        sig, sent = (signal.SIGKILL, kill_sent) if elapsed >= timeout / 2 else (signal.SIGTERM, term_sent)
        for pid, record in tree.items():
            identity = (pid, record['start'])
            if identity not in sent:
                # Check the identity again right before signalling, against PID reuse.
                current = process_table().get(pid)
                if current and current['start'] == record['start']:
                    try:
                        os.kill(pid, sig)
                    except ProcessLookupError:
                        pass
                sent.add(identity)
        state = query_gpu(gpu)
        last = {'live_pids': sorted(tree), 'gpu': state, 'elapsed_s': elapsed}
        if not tree and process.poll() is not None and state['uuid'] == expected_uuid \
                and not state['compute_pids'] and state['memory_mib'] < 100:
            return last
        if elapsed >= timeout:
            raise ShutdownError('Server tree or GPU memory did not release within timeout: ' + repr(last))
        time.sleep(min(.25, timeout - elapsed))


def gpu_placement(pid, gpu, expected_uuid):
    """After readiness, the server's compute processes are on the given GPU and nothing else is."""
    owned = processes(pid)
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
                                   '--format=csv,noheader,nounits'], text=True, timeout=5)
    apps = [{'pid': int(p), 'uuid': uuid.strip()} for p, uuid in csv.reader(
        io.StringIO(raw), skipinitialspace=True) if p.strip()]
    ours = [a for a in apps if a['pid'] in owned]
    if {a['uuid'] for a in ours} != {expected_uuid}:
        raise RuntimeError(f'Server compute processes must occupy exactly GPU {gpu}')
    if any(a['uuid'] == expected_uuid and a['pid'] not in owned for a in apps):
        raise RuntimeError(f'Foreign compute process on GPU {gpu} after readiness')
    return {'gpu_index': gpu, 'uuid': expected_uuid, 'compute_processes': ours,
            'server_tree_pids': sorted(owned)}


# ---------------------------------------------------------------- environment

def reject_overrides(env):
    """Inherited variables that would change what the engine does stop the run."""
    overrides = [k for k in env if (
        k.startswith(('VLLM_', 'NCCL_', 'PYTORCH_', 'OMP_', 'TRITON_', 'FLASHINFER_', 'CUBLAS_',
                      'CUBLASLT_', 'CUDNN_', 'NVIDIA_TF32_', 'MKL_', 'OPENBLAS_'))
        or (k.startswith('TORCH') and k != 'TORCH_HOME')
        or (k.startswith('CUDA_') and k != 'CUDA_DEVICE_ORDER')
        or k in ('LD_PRELOAD', 'LD_LIBRARY_PATH', 'PYTHONPATH'))]
    if env.get('CUDA_DEVICE_ORDER', 'PCI_BUS_ID') != 'PCI_BUS_ID':
        overrides.append('CUDA_DEVICE_ORDER')
    if overrides:
        raise RuntimeError('Unexpected runtime overrides: ' + ', '.join(sorted(overrides)))


def sampling_observation(log):
    lines = [line for line in log.splitlines() if
             'Default vLLM sampling parameters have been overridden by' in line]
    values = [ast.literal_eval(re.search(r': `(\{.*\})`\.', line).group(1)) for line in lines]
    if not values or any(v != values[0] for v in values):
        raise RuntimeError('Missing or conflicting server default-sampling-params log')
    return {'default_sampling_params_log_lines': lines, 'default_sampling_params': values[0]}


SECRET_WORDS = ('KEY', 'TOKEN', 'SECRET', 'PASSWORD', 'CREDENTIAL', 'AUTH')


def redact(env):
    """A copy of an environment fit to record: values of secret-like keys are masked."""
    def secret(key):
        upper = key.upper()
        return upper.startswith('SSH_') or any(word in upper for word in SECRET_WORDS)
    return {k: ('<redacted>' if secret(k) else v) for k, v in env.items()}


def vllm_binary():
    return Path(sys.executable).parent / 'vllm'


def check_versions():
    """vLLM 0.30.0 (CUDA 12.9 build, commit ced6857a) with a CUDA 12.9 torch."""
    if importlib.metadata.version('vllm') != VLLM_VERSION:
        raise RuntimeError(f'vllm=={VLLM_VERSION} required')
    torch = importlib.metadata.distribution('torch')
    tree = ast.parse(Path(torch.locate_file('torch/version.py')).read_text())
    cuda = None
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == 'cuda':
            cuda = ast.literal_eval(node.value)
    if cuda != TORCH_CUDA:
        raise RuntimeError(f'A torch built for CUDA {TORCH_CUDA} is required')
    distribution = importlib.metadata.distribution('vllm')
    version_text = Path(distribution.locate_file('vllm/_version.py')).read_text()
    if COMMIT[:9] not in version_text:
        raise RuntimeError('Installed vLLM was not built from commit ' + COMMIT[:9])
    if not vllm_binary().is_file():
        raise RuntimeError(f'No vllm executable next to {sys.executable}')


def environment(cache_dir, gpu):
    """The server's environment: inherited, checked, plus device order, offline hub and caches."""
    check_versions()
    reject_overrides(os.environ)
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               VLLM_CACHE_ROOT=str(cache / 'vllm'), TRITON_CACHE_DIR=str(cache / 'triton'),
               CUDA_CACHE_PATH=str(cache / 'cuda'), XDG_CACHE_HOME=str(cache / 'xdg'))
    return env


def parse_cpulist(text):
    cpus = set()
    for part in text.strip().split(','):
        if part:
            lo, _, hi = part.partition('-')
            cpus.update(range(int(lo), int(hi or lo) + 1))
    return cpus


def topology(gpu, lscpu=None, bus=None, sysfs=Path('/sys')):
    """Server on the GPU's socket; client on the first four physical cores of another socket."""
    raw = lscpu if lscpu is not None else subprocess.check_output(
        ['lscpu', '-p=CPU,CORE,SOCKET,ONLINE'], text=True)
    cores = {}
    for line in raw.splitlines():
        if line.startswith('#'):
            continue
        cpu, core, sock, online = line.split(',')
        if online == 'Y':
            cores.setdefault((int(sock), int(core)), []).append(int(cpu))
    socket_ids = sorted({s for s, _ in cores})
    if len(socket_ids) < 2:
        raise RuntimeError('Two CPU sockets are required')
    sockets = {s: sorted(c for (sock, _), cpus in cores.items() if sock == s for c in cpus) for s in socket_ids}
    if bus is None:
        bus = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=pci.bus_id',
                                       '--format=csv,noheader'], text=True).strip()
    bus = bus.lower()
    domain, rest = bus.split(':', 1)
    node = int((sysfs / 'bus/pci/devices' / f'{int(domain, 16):04x}:{rest}' / 'numa_node').read_text())
    if node < 0:
        raise RuntimeError('The GPU reports no NUMA node')
    node_cpus = parse_cpulist((sysfs / 'devices/system/node' / f'node{node}' / 'cpulist').read_text())
    server = [s for s in socket_ids if node_cpus <= set(sockets[s])]
    if len(server) != 1:
        raise RuntimeError("The GPU's NUMA node does not lie within one socket")
    server = server[0]
    client = min(s for s in socket_ids if s != server)
    first = [cpus for (sock, _), cpus in sorted(cores.items()) if sock == client][:4]
    if len(first) != 4:
        raise RuntimeError('Four physical client cores are not available')
    return {'gpu': gpu, 'client': [c for group in first for c in group], 'server': sockets[server],
            'client_socket': str(client), 'server_socket': str(server),
            'sockets': {str(client): sockets[client], str(server): sockets[server]},
            'lscpu': raw, 'numa_node': node, 'pci_bus': bus}


class Locks:
    """Exclusive flocks held for the whole run and inherited by each server."""
    def __init__(self, paths):
        self.handles = []
        try:
            for p in paths:
                p = Path(p)
                p.parent.mkdir(parents=True, exist_ok=True)
                h = p.open('a+')
                self.handles.append(h)
                fcntl.flock(h, fcntl.LOCK_EX | fcntl.LOCK_NB)
                h.seek(0)
                h.truncate()
                h.write(str(os.getpid()) + '\n')
                h.flush()
        except BaseException:
            self.close()
            raise

    @property
    def fds(self):
        return tuple(h.fileno() for h in self.handles)

    def close(self):
        # No LOCK_UN: a server that inherited the descriptor keeps the lock if this process dies.
        for h in self.handles:
            h.close()
        self.handles.clear()


def argv(arm, port, cpus, model):
    command = ['taskset', '-c', ','.join(map(str, cpus)), str(vllm_binary()),
               'serve', str(model), '--port', str(port)]
    kv = config(arm)
    if kv:
        command += ['--kv-transfer-config', json.dumps(kv, separators=(',', ':'))]
    return command


# ---------------------------------------------------------------- start-test receipts

def verify_pinning_start(arm, start):
    """Re-read the DEBUG start's log and launch record, not only a stored pass flag."""
    from .gates import PINNED_OK, pinning_observation
    if (start.get('arm') != arm or start.get('debug_pinning') is not True
            or start.get('pinning') != PINNED_OK):
        raise RuntimeError('A successful separate DEBUG start is required for ' + arm)
    receipt = start.get('pinning_receipt', {})
    paths = {}
    for kind in ('log', 'launch'):
        path = Path(receipt.get(kind + '_path', '/nonexistent'))
        if not path.is_file() or digest(path) != receipt.get(kind + '_sha256'):
            raise RuntimeError(f'DEBUG start {kind} of {arm} is missing or changed')
        paths[kind] = path
    launch = json.loads(paths['launch'].read_text())
    if (launch.get('arm') != arm or launch.get('debug_pinning') is not True
            or launch.get('environment', {}).get('VLLM_LOGGING_LEVEL') != 'DEBUG'):
        raise RuntimeError('DEBUG start record does not identify this arm and DEBUG logging')
    command = launch.get('argv', [])
    if (str(vllm_binary()) not in command or command.count('--kv-transfer-config') != 1
            or json.loads(command[command.index('--kv-transfer-config') + 1]) != config(arm)):
        raise RuntimeError('DEBUG start did not run this arm with this vLLM installation')
    if pinning_observation(paths['log'].read_text()) != PINNED_OK:
        raise RuntimeError('DEBUG start log does not show a successful registration')


def run_identity(args):
    """What a start test and the timed run must share: GPU, model, compile caches, interpreter."""
    return {'gpu': args.gpu, 'model': str(Path(args.model).resolve()),
            'cache_dir': str(Path(args.cache_dir).resolve()), 'python': sys.executable}


def verify_start_tests(path, sessions_path, args):
    """The timed run accepts only a start test made by this code, with these sessions, on the same
    GPU, model directory, cache directory and interpreter."""
    path = Path(path)
    artifact = json.loads(path.read_text())
    starts = artifact.get('starts', {})
    if set(starts) != set(ARMS) or any(type(s.get('pass')) is not bool for s in starts.values()):
        raise RuntimeError('Start test does not account for all five arms')
    if (artifact.get('mode') != 'start-test' or artifact.get('code_sha256') != code_hashes()
            or artifact.get('sessions_sha256') != digest(sessions_path)):
        raise RuntimeError('Start test was made by other code or with other sessions')
    manifest_path = path.parent / 'manifest.json'
    if not manifest_path.is_file():
        raise RuntimeError('Start test manifest.json not found next to ' + path.name)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('code_sha256') != code_hashes():
        raise RuntimeError('Start test manifest was written by other code')
    recorded, current = manifest.get('identity') or {}, run_identity(args)
    different = sorted(k for k in current if recorded.get(k) != current[k])
    if different:
        raise RuntimeError('Start test used a different ' + ', '.join(
            f'{k} ({recorded.get(k)!r}, now {current[k]!r})' for k in different))
    for arm in NATIVE_ARMS:
        if starts[arm]['pass']:
            verify_pinning_start(arm, starts[arm])
    return {'path': str(path.resolve()), 'sha256': digest(path), 'starts': starts}


# ---------------------------------------------------------------- server

class Server:
    def __init__(self, arm, args, topo, output, locks, *, debug_pinning=False):
        if debug_pinning and arm not in NATIVE_ARMS:
            raise ValueError('A DEBUG start is only made for the two native arms')
        self.directory = output
        self.gpu = args.gpu
        output.mkdir()
        env = environment(args.cache_dir, args.gpu)
        if debug_pinning:
            env['VLLM_LOGGING_LEVEL'] = 'DEBUG'
        idle = query_gpu(args.gpu)
        self.gpu_uuid = idle['uuid']
        if idle['compute_pids'] or idle['memory_mib'] >= 100:
            raise RuntimeError(f'GPU {args.gpu} is occupied; no launch')
        # An occupied port must never be mistaken for this server's health endpoint.
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(('127.0.0.1', args.port))
        self.model = Path(args.model)
        self.argv = argv(arm, args.port, topo['server'], self.model)
        generation_path = self.model / 'generation_config.json'
        generation = {'path': str(generation_path), 'sha256': digest(generation_path),
                      'contents': json.loads(generation_path.read_text())}
        self.port = args.port
        self.log = (output / 'server.log').open('x')
        try:
            self.process = subprocess.Popen(self.argv, stdin=subprocess.DEVNULL, stdout=self.log,
                stderr=subprocess.STDOUT, env=env, start_new_session=True, pass_fds=locks.fds)
        except BaseException:
            self.log.close()
            raise
        self.known_processes = {}
        self.stopped = False
        try:
            owned_tree(self.process.pid, self.known_processes)
            atomic_json(output / 'launch.json', {'argv': self.argv, 'pid': self.process.pid,
                'arm': arm, 'debug_pinning': debug_pinning,
                'wall_time': time.time(), 'topology': topo, 'idle': idle,
                'generation_config': generation, 'python': sys.executable,
                'versions': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
                'environment': redact(env)})
        except BaseException as error:
            self.retain(error)
            raise

    def ready(self):
        deadline = time.monotonic() + 1200
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError('Server exited before readiness')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{self.port}/health', timeout=2) as response:
                    if response.status == 200 and self.process.poll() is None:
                        break
            except OSError:
                pass
            time.sleep(1)
        else:
            raise TimeoutError('Server start timed out')
        # Failures from here on end this start; they are not readiness retries.
        tree = owned_tree(self.process.pid, self.known_processes)
        inodes = listener_ownership(self.port, tree)
        with urllib.request.urlopen(f'http://127.0.0.1:{self.port}/v1/models', timeout=5) as response:
            models = json.load(response)
        if {m['id'] for m in models.get('data', [])} != {str(self.model)}:
            raise RuntimeError('Answering server reports a different model path')
        placement = gpu_placement(self.process.pid, self.gpu, self.gpu_uuid)
        if self.process.poll() is not None:
            raise RuntimeError('Server exited during readiness verification')
        manifest = json.loads((self.directory / 'launch.json').read_text())
        manifest.update(gpu_placement=placement, listener_inodes=inodes, models=models)
        manifest.update(sampling_observation((self.directory / 'server.log').read_text()))
        atomic_json(self.directory / 'launch.json', manifest)

    def completed(self):
        if self.stopped:
            return
        try:
            receipt = stop_server(self.process, self.known_processes, self.gpu, self.gpu_uuid, timeout=60)
            atomic_json(self.directory / 'shutdown.json', receipt)
            self.stopped = True
        except Exception as error:
            atomic_json(self.directory / 'shutdown-failure.json', {'error': repr(error)})
            raise ShutdownError(str(error)) from error
        finally:
            self.log.close()

    def retain(self, error):
        atomic_json(self.directory / 'failure-process.json', {
            'error': repr(error), 'server_pid': self.process.pid, 'coordinator_pid': os.getpid(),
            'argv': self.argv, 'action': 'Bounded cleanup; the run stops if release cannot be proved.'})
        self.completed()
