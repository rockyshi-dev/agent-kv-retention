"""Metric parsing, request isolation, the stream reader and the eviction filler."""
import io
import json
import unittest
from unittest.mock import Mock, patch

from akvr import client, gates
from akvr.spec import ARMS

from .helpers import fixture, metric_text


class Metrics(unittest.TestCase):
    def test_real_scrape_parses_and_sums_labelled_series(self):
        raw = fixture('metrics_start.txt')
        values = client.parse_metrics(raw)
        self.assertEqual(values['request_success'], 0)
        self.assertIsNone(values['kv_offload_load_bytes'])
        cfg = client.cache_config(raw)
        self.assertEqual((cfg['num_gpu_blocks'], cfg['block_size'], cfg['enable_prefix_caching']),
                         ('7380', '16', 'True'))
        two = metric_text(3).replace('vllm:request_success_total{model_name="m",engine="0"} 3',
                                     'vllm:request_success_total{reason="stop"} 2\nvllm:request_success_total{reason="length"} 1')
        self.assertEqual(client.parse_metrics(two)['request_success'], 3)

    def test_missing_reset_and_foreign_requests_rejected(self):
        a = client.parse_metrics(metric_text())
        b = client.parse_metrics(metric_text(1))
        self.assertEqual(client.delta(a, b)['request_success'], 1)
        with self.assertRaises(ValueError):
            client.parse_metrics('')
        with self.assertRaises(ValueError):
            client.delta(b, a)
        with self.assertRaises(ValueError):
            client.delta(a, client.parse_metrics(metric_text(2)))
        with self.assertRaises(ValueError):
            client.delta(a, client.parse_metrics(metric_text(1, running=1)))

    def test_native_load_counter_required_after_first_resume(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                raw = metric_text()
                c = client.Client(8066, 'm', arm=arm)
                response = Mock(status=200)
                response.read.return_value = raw.encode()
                conn = Mock()
                conn.getresponse.return_value = response
                try:
                    with patch.object(client.http.client, 'HTTPConnection', return_value=conn):
                        # vLLM creates the counter lazily, on the first load from host memory.
                        self.assertIsNone(c.metrics()['values']['kv_offload_load_bytes'])
                        c.load_required = True
                        if arm in ('NATIVE', 'NATIVE-ALL'):
                            with self.assertRaisesRegex(ValueError, 'kv_offload_load_bytes'):
                                c.metrics()
                        else:
                            self.assertIsNone(c.metrics()['values']['kv_offload_load_bytes'])
                    before = client.parse_metrics(metric_text(load=0), arm)
                    after = client.parse_metrics(metric_text(1, load=57344), arm)
                    self.assertEqual(client.delta(before, after)['kv_offload_load_bytes'], 57344)
                finally:
                    c.close()


class Requests(unittest.TestCase):
    def test_idle_completion_connection_is_replaced_before_request(self):
        c = client.Client(8066, 'm')
        old = Mock(sock=True)
        c.conn, c.last_used_at = old, 0
        new = Mock(sock=None)
        new.getresponse.return_value = Mock(status=200)
        new.getresponse.return_value.read.return_value = json.dumps({
            'choices': [{'token_ids': [7]}], 'usage': {'prompt_tokens': 1}}).encode()
        before = {'values': client.parse_metrics(metric_text())}
        after = {'values': client.parse_metrics(metric_text(1, 1))}
        with patch.object(client.http.client, 'HTTPConnection', return_value=new), \
             patch.object(c, 'metrics', return_value=before), patch.object(c, 'fence', return_value=after):
            result = c.request([9], 1, force=True)
        old.close.assert_called_once()
        old.putrequest.assert_not_called()
        new.connect.assert_called_once()
        self.assertEqual(result['payload']['min_tokens'], 1)
        self.assertTrue(result['payload']['ignore_eos'])
        self.assertTrue(result['payload']['return_token_ids'])
        self.assertEqual(result['payload']['temperature'], 0)
        c.close()

    def test_stream_empty_text_token_is_ttft_and_final_usage_survives(self):
        chunks = [
            {'choices': [{'text': '', 'token_ids': [999]}]},
            {'choices': [{'text': 'x', 'token_ids': [5]}]},
            {'choices': [], 'usage': {'prompt_tokens': 2048}},
        ]
        raw = b''.join(('data: ' + json.dumps(ch) + '\n\n').encode() for ch in chunks) + b'data: [DONE]\n\n'
        response = io.BytesIO(raw)
        response.status = 200
        c = client.Client(8066, 'm')
        c.conn = Mock(sock=True)
        c.conn.getresponse.return_value = response
        a = {'values': client.parse_metrics(metric_text()), 'raw': metric_text()}
        b = {'values': client.parse_metrics(metric_text(1, 2048)), 'raw': metric_text(1, 2048)}
        with patch.object(c, 'metrics', return_value=a), patch.object(c, 'fence', return_value=b):
            result = c.request([17] * 2048, 8, stream=True)
        self.assertGreater(result['ttft_ms'], 0)
        self.assertEqual(client.first_token_ms(result), result['ttft_ms'])
        self.assertNotIn('return_token_ids', result['payload'])
        self.assertEqual(result['payload']['stream_options'], {'include_usage': True})
        self.assertEqual(result['usage']['prompt_tokens'], 2048)

    def test_admission_counter_must_match_the_prompt(self):
        c = client.Client(8066, 'm')
        c.conn = Mock(sock=True)
        c.conn.getresponse.return_value = Mock(status=200)
        c.conn.getresponse.return_value.read.return_value = json.dumps({
            'choices': [{'token_ids': [7]}], 'usage': {'prompt_tokens': 4}}).encode()
        before = {'values': client.parse_metrics(metric_text())}
        after = {'values': client.parse_metrics(metric_text(1, 5))}
        with patch.object(c, 'metrics', return_value=before), patch.object(c, 'fence', return_value=after):
            with self.assertRaisesRegex(ValueError, 'Admission'):
                c.request([1, 2, 3, 4], 1)
        c.close()

    def test_foreign_admission_cannot_supply_our_host_hits(self):
        before = {'values': client.parse_metrics(metric_text())}
        after = {'values': client.parse_metrics(metric_text(1, 4096, external=1776, running=1))}
        c = client.Client(8066, 'm')
        with patch.object(c, 'metrics', return_value=after):
            with self.assertRaises(RuntimeError):
                c.fence(before, timeout=0)
        with self.assertRaises(ValueError):
            client.delta(before['values'], after['values'])
        with patch.object(c, 'metrics', return_value={'values': client.parse_metrics(metric_text(2))}):
            with self.assertRaisesRegex(RuntimeError, 'overlapping'):
                c.fence(before, timeout=0)
        c.close()


class FillerTests(unittest.TestCase):
    def test_filler_overshoots_and_never_reuses_first_block(self):
        f = client.Filler(list(range(100, 400)), [[100] * 16])
        c = Mock()
        c.request.return_value = {'usage': {'prompt_tokens': 4096},
                                  'delta': {'prefix_cache_hits': 0, 'external_prefix_cache_hits': 0}}
        written = []
        tokens = f.evict(c, 8192, written.append)
        # The total must exceed 1.5 x 8192 = 12288: four prompts of 4096.
        self.assertEqual(tokens, 16384)
        self.assertEqual(len(written), 4)
        heads = [tuple(call.args[0][:16]) for call in c.request.call_args_list]
        self.assertEqual(len(set(heads)), 4)
        self.assertNotIn(tuple([100] * 16), heads)
        self.assertTrue(all(len(call.args[0]) == 4096 and call.args[1] == 1 for call in c.request.call_args_list))

    def test_filler_that_hits_the_cache_stops_the_session(self):
        f = client.Filler(list(range(100, 400)), [])
        c = Mock()
        c.request.return_value = {'usage': {'prompt_tokens': 4096},
                                  'delta': {'prefix_cache_hits': 16, 'external_prefix_cache_hits': 0}}
        with self.assertRaisesRegex(ValueError, 'reused'):
            f.evict(c, 8192, lambda _: None)


class SessionGates(unittest.TestCase):
    def test_counts_from_one_session(self):
        row = {'arm': 'NATIVE', 'N': 2048, 'warmup': False, 'turn1': {'token_ids': [1] * 512},
               'resume': {'usage': {'prompt_tokens': 2048}, 'ttft_ms': 100.0,
                          'delta': {'request_success': 1, 'prefix_cache_hits': 0,
                                    'external_prefix_cache_hits': 1280, 'kv_offload_load_bytes': 1280 * 57344}}}
        g = gates.session_gates(row)
        self.assertEqual((g['L'], g['E'], g['H'], g['G'], g['Hh'], g['R']), (True, True, True, 0, 1280, 768))
        self.assertFalse(g['load_mismatch'])
        row['resume']['usage']['prompt_tokens'] = 2047
        self.assertFalse(gates.session_gates(row)['L'])


if __name__ == '__main__':
    unittest.main()
