"""Sequential /v1/completions client; every request is bracketed by /metrics snapshots."""
import http.client
import json
import math
import re
import time

from .spec import FILLER_FACTOR, FILLER_PROMPT

COUNTERS = ('prefix_cache_queries', 'prefix_cache_hits', 'external_prefix_cache_hits', 'request_success')
GAUGES = ('num_requests_running', 'num_requests_waiting')
NATIVE_ARMS = ('NATIVE', 'NATIVE-ALL')


def samples(raw, name):
    return re.findall(r'^' + re.escape('vllm:' + name) + r'(?:\{([^\n]*)\})?\s+(\S+)\s*$', raw, re.M)


def counter(raw, name, required=True):
    values = samples(raw, name + '_total')
    if not values:
        if required:
            raise ValueError('Missing metric: ' + name)
        return None
    value = sum(float(x[1]) for x in values)
    if not math.isfinite(value) or value < 0:
        raise ValueError('Invalid metric: ' + name)
    return value


def parse_metrics(raw, arm=None):
    """Counters and gauges of one scrape. `arm` set to a native arm requires the load counter."""
    values = {name: counter(raw, name) for name in COUNTERS} | {
        'kv_offload_load_bytes': counter(raw, 'kv_offload_load_bytes', required=arm in NATIVE_ARMS)}
    for name in GAUGES:
        rows = samples(raw, name)
        if not rows:
            raise ValueError('Missing scheduler gauge: ' + name)
        value = sum(float(v) for _, v in rows)
        if not math.isfinite(value) or value < 0 or value != int(value):
            raise ValueError('Invalid scheduler gauge: ' + name)
        values[name] = value
    return values


def cache_config(raw):
    rows = samples(raw, 'cache_config_info')
    if len(rows) != 1 or float(rows[0][1]) != 1:
        raise ValueError('Expected exactly one cache configuration series')
    return {k: json.loads(v) for k, v in re.findall(r'(\w+)=("(?:[^"\\]|\\.)*")', rows[0][0])}


def delta(before, after):
    """Counter differences of one request; nothing else may be running at either scrape."""
    if any(before[name] != 0 or after[name] != 0 for name in GAUGES):
        raise ValueError('Request metrics bracket an active or waiting foreign request')
    result = {}
    for key, old in before.items():
        new = after[key]
        if old is None or new is None:
            result[key] = None
        elif new < old:
            raise ValueError('Counter reset during a session: ' + key)
        else:
            result[key] = new - old
    if result['request_success'] != 1:
        raise ValueError('Metrics do not isolate exactly one completed request')
    return result


def first_token_ms(record):
    """TTFT: send time to the first stream chunk that carries a choice.

    vLLM v0.30.0 emits a chunk with a choice only once a token exists, so the first such chunk
    carries the first token, also when its text is empty.
    """
    for event in record.get('events', []):
        if event['chunk'].get('choices'):
            return (event['arrival_ns'] - record['start_ns']) / 1e6
    return None


class Client:
    def __init__(self, port, model, arm=None):
        self.port, self.model = port, model
        self.arm = arm
        # vLLM v0.30.0 creates the native load counter only on the first actual load from host
        # memory, so it is required from the end of a start's first resume (a warm-up) on.
        self.load_required = False
        self.conn = http.client.HTTPConnection('127.0.0.1', port, timeout=300)
        self.last_used_at = None
        self.last_record = None
        self.last_metrics_raw = None

    def close(self):
        self.conn.close()

    def metrics(self):
        conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=10)
        try:
            conn.request('GET', '/metrics')
            response = conn.getresponse()
            raw = response.read().decode()
            self.last_metrics_raw = raw
            if response.status != 200:
                raise RuntimeError('Metrics HTTP error')
            return {'raw': raw, 'values': parse_metrics(raw, self.arm if self.load_required else None),
                    'time_ns': time.monotonic_ns()}
        finally:
            conn.close()

    def fence(self, before, timeout=30):
        """Wait until exactly one more request has completed and nothing is running."""
        target = before['values']['request_success'] + 1
        end = time.monotonic() + timeout
        while True:
            after = self.metrics()
            if self.last_record is not None:
                self.last_record['last_fence_snapshot'] = after
            count = after['values']['request_success']
            if count > target:
                raise RuntimeError('Foreign request or overlapping request detected')
            if count == target and all(after['values'][name] == 0 for name in GAUGES):
                return after
            if time.monotonic() >= end:
                raise RuntimeError('Request completion metrics did not arrive')
            time.sleep(0.01)

    def request(self, ids, output, stream=False, force=False):
        payload = {'model': self.model, 'prompt': ids, 'max_tokens': output,
                   'temperature': 0, 'stream': stream}
        if not stream:
            # Turn 1 and filler. The streamed resume request does not ask for ids: with them its
            # first chunk would also carry all N prompt ids, inside TTFT.
            payload['return_token_ids'] = True
        if force:
            payload.update(min_tokens=output, ignore_eos=True)
        if stream:
            payload['stream_options'] = {'include_usage': True}
        # All encoding happens before the clock starts at the first write of the request.
        body = json.dumps(payload, separators=(',', ':')).encode()
        before = self.metrics()
        result = {'before': before, 'prompt_length': len(ids), 'token_ids': [], 'events': [],
                  'usage': None, 'ttft_ms': None, 'payload': payload}
        self.last_record = result
        if any(before['values'][name] != 0 for name in GAUGES):
            raise RuntimeError('Server has running or waiting requests before this request')
        if self.last_used_at is not None and time.monotonic() - self.last_used_at > 1:
            self.conn.close()
            self.conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=300)
        if self.conn.sock is None:
            self.conn.connect()
        self.conn.putrequest('POST', '/v1/completions')
        self.conn.putheader('Content-Type', 'application/json')
        self.conn.putheader('Content-Length', str(len(body)))
        started = time.perf_counter_ns()
        result['start_ns'] = started
        self.conn.endheaders(body)
        response = self.conn.getresponse()
        if response.status != 200:
            raise RuntimeError(f'Completion HTTP {response.status}: {response.read().decode()}')
        if not stream:
            value = json.loads(response.read())
            result['response'] = value
            result['token_ids'] = value['choices'][0].get('token_ids')
            result['usage'] = value.get('usage')
        else:
            lines, done = [], False
            while True:
                line = response.readline()
                now = time.perf_counter_ns()
                if not line:
                    break
                if line.strip():
                    lines.append(line.decode().rstrip('\r\n'))
                    continue
                data = '\n'.join(x[5:].lstrip() for x in lines if x.startswith('data:'))
                lines = []
                if not data:
                    continue
                if data == '[DONE]':
                    done = True
                    continue
                value = json.loads(data)
                if value.get('error'):
                    raise RuntimeError(str(value['error']))
                result['events'].append({'arrival_ns': now, 'chunk': value})
                if value.get('usage') is not None:
                    result['usage'] = value['usage']
                if value.get('choices') and result['ttft_ms'] is None:
                    result['ttft_ms'] = (now - started) / 1e6
            if not done or result['ttft_ms'] is None:
                raise ValueError('Incomplete SSE or no emitted token')
        result['end_ns'] = time.perf_counter_ns()
        self.last_used_at = time.monotonic()
        result['after'] = self.fence(before)
        result['delta'] = delta(before['values'], result['after']['values'])
        if result['delta']['prefix_cache_queries'] != len(ids):
            raise ValueError('Admission token counter does not isolate this prompt')
        return result


class Filler:
    """Distinct 4,096-token prompts; no two share a first cache block, nor with any session."""
    def __init__(self, vocabulary, forbidden):
        self.vocab = vocabulary
        self.seen = set(map(tuple, forbidden))
        self.serial = 0

    def next(self):
        base = len(self.vocab)
        while True:
            value = self.serial
            self.serial += 1
            if value >= base ** 16:
                raise RuntimeError('Filler namespace exhausted')
            first = []
            for _ in range(16):
                first.append(self.vocab[value % base])
                value //= base
            key = tuple(first)
            if key not in self.seen:
                self.seen.add(key)
                return first + [self.vocab[i % base] for i in range(FILLER_PROMPT - 16)]

    def evict(self, client, pool_tokens, write):
        """Send fillers one after another until their total exceeds 1.5 x the GPU KV pool."""
        tokens = 0
        while tokens <= FILLER_FACTOR * pool_tokens:
            ids = self.next()
            result = client.request(ids, 1, force=True)
            if result['usage']['prompt_tokens'] != FILLER_PROMPT:
                raise ValueError('Filler token length mismatch')
            if result['delta']['prefix_cache_hits'] != 0 or result['delta']['external_prefix_cache_hits'] != 0:
                raise ValueError('Filler unexpectedly reused a cached prefix')
            write(result)
            tokens += FILLER_PROMPT
        return tokens
