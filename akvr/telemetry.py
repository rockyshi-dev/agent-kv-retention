"""One-second CPU and GPU observations for the quiet check. No CUDA imports."""
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import threading
import time


def query_gpu(gpu):
    row = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=uuid,clocks.sm,power.draw,temperature.gpu,utilization.gpu,memory.used',
        '--format=csv,noheader,nounits'], text=True, timeout=5)
    values = next(csv.reader(io.StringIO(row), skipinitialspace=True))
    pids = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid',
                                    '--format=csv,noheader,nounits'], text=True, timeout=5)
    return {'uuid': values[0], 'clock_mhz': float(values[1]), 'power_w': float(values[2]),
            'temperature_c': float(values[3]), 'utilization_pct': float(values[4]),
            'memory_mib': float(values[5]), 'compute_pids': [int(x) for x in pids.splitlines() if x.strip()]}


def processes(root, proc=Path('/proc')):
    """`root` and all its descendants, with their CPU ticks."""
    table = {}
    for p in proc.iterdir():
        if not p.name.isdigit():
            continue
        try:
            fields = (p / 'stat').read_text().rsplit(')', 1)[1].split()
            table[int(p.name)] = {'parent': int(fields[1]), 'start': int(fields[19]),
                                  'ticks': sum(int(x) for x in fields[11:15])}
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    owned = {root}
    while True:
        new = {p for p, v in table.items() if v['parent'] in owned}
        if new <= owned:
            break
        owned |= new
    if root not in table:
        raise RuntimeError('Coordinator absent')
    return {p: table[p] for p in owned if p in table}


def cpu_times(proc=Path('/proc')):
    cpu = {}
    for line in (proc / 'stat').read_text().splitlines():
        f = line.split()
        if f[0].startswith('cpu') and f[0][3:].isdigit():
            v = list(map(int, f[1:9]))
            cpu[f[0][3:]] = {'total': sum(v), 'busy': sum(v) - v[3] - v[4]}
    return cpu


def capture(root, topo, server_pid):
    """One sample: owned CPU ticks per socket, per-CPU times, thread affinities, GPU state."""
    client_socket, server_socket = topo['client_socket'], topo['server_socket']
    sockets = topo['sockets']
    procs = processes(root)
    server_tree = processes(server_pid)
    ticks = {client_socket: 0, server_socket: 0}
    placement, affinities = True, {}
    for pid, record in procs.items():
        socket = server_socket if pid in server_tree else client_socket
        ticks[socket] += record['ticks']
        try:
            for task in (Path('/proc') / str(pid) / 'task').iterdir():
                affinity = sorted(os.sched_getaffinity(int(task.name)))
                affinities[task.name] = affinity
                placement &= bool(affinity) and set(affinity) <= set(sockets[socket])
        except (FileNotFoundError, ProcessLookupError):
            continue
    cpu = cpu_times()  # read right after the owned ticks, before the slower GPU query
    gpu = query_gpu(topo['gpu'])
    gpu['foreign_pids'] = sorted(set(gpu['compute_pids']) - set(procs))
    return {'time': time.monotonic(), 'wall_time': time.time(), 'cpu': cpu, 'gpu': gpu,
            'owned_ticks': ticks, 'processes': procs, 'placement_ok': placement,
            'thread_affinities': affinities}


class Recorder:
    def __init__(self, path, topo, server_pid):
        self.path, self.topo, self.server_pid = path, topo, server_pid
        self.root = os.getpid()
        self.stop_event = threading.Event()
        self.samples = []
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def take(self, f):
        try:
            row = capture(self.root, self.topo, self.server_pid)
        except Exception as e:
            row = {'time': time.monotonic(), 'error': repr(e)}
        if self.samples and 'cpu' in row and 'cpu' in self.samples[-1]:
            previous = self.samples[-1]['cpu']
            row['cpu_utilization_pct'] = {}
            for cpu, current in row['cpu'].items():
                old = previous.get(cpu)
                dt = current['total'] - old['total'] if old else 0
                row['cpu_utilization_pct'][cpu] = 100 * (current['busy'] - old['busy']) / dt if dt > 0 else None
        self.samples.append(row)
        f.write(json.dumps(row) + '\n')
        f.flush()
        self.ready.set()

    def run(self):
        try:
            with self.path.open('x') as f:
                deadline = time.monotonic()
                while not self.stop_event.is_set():
                    self.take(f)
                    deadline += 1
                    self.stop_event.wait(max(0, deadline - time.monotonic()))
                self.take(f)
        finally:
            self.ready.set()

    def start(self):
        self.thread.start()
        self.ready.wait(10)
        if not self.samples:
            raise RuntimeError('Telemetry did not start')

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=15)
        if self.thread.is_alive():
            raise RuntimeError('Telemetry did not stop')
        return self.samples
