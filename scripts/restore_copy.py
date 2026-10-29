"""Host-to-GPU copy time of every restored resume request, from the /metrics text that `run`
stores before and after each request (vllm:kv_offload_load_bytes_total and
vllm:kv_offload_load_time_total, the OffloadingConnector's own timer).

Usage: python scripts/restore_copy.py RUN/raw.json

Prints, per configuration and context length, the median copy time, bytes and rate over the
measured sessions of the rounds `report` uses, and the time per copied block-layer (the connector
keeps one host tensor per layer, so one 16-token block of one layer is one copy)."""
import json
import statistics
import sys

BLOCK = 16
LAYERS = 28          # Qwen2.5-7B-Instruct
BYTES_PER_TOKEN = 57344


def metric(text, name):
    return sum(float(line.rsplit(' ', 1)[1]) for line in text.splitlines() if line.startswith(name + '{'))


def main(path):
    raw = json.load(open(path))
    cells = {}
    for rnd in raw['rounds']:
        if not rnd['selected']:
            continue
        for s in rnd['sessions']:
            if s['warmup'] or s['arm'] not in ('NATIVE', 'NATIVE-ALL'):
                continue
            before, after = s['resume']['before']['raw'], s['resume']['after']['raw']
            nbytes = metric(after, 'vllm:kv_offload_load_bytes_total') - metric(before, 'vllm:kv_offload_load_bytes_total')
            secs = metric(after, 'vllm:kv_offload_load_time_total') - metric(before, 'vllm:kv_offload_load_time_total')
            cells.setdefault((s['arm'], s['N']), []).append((nbytes, secs))
    print(f"{'cell':18} {'n':>3} {'copy ms':>8} {'GB':>6} {'GB/s':>6} {'copies':>7} {'us/copy':>8}")
    for (arm, n), rows in sorted(cells.items()):
        ms = statistics.median(t for _, t in rows) * 1e3
        nbytes = statistics.median(b for b, _ in rows)
        copies = round(nbytes / BYTES_PER_TOKEN / BLOCK) * LAYERS
        print(f"{arm + '/' + str(n):18} {len(rows):3} {ms:8.1f} {nbytes / 1e9:6.2f} {nbytes / ms / 1e6:6.2f} "
              f"{copies:7d} {ms * 1e3 / copies:8.2f}")


if __name__ == '__main__':
    main(sys.argv[1])
