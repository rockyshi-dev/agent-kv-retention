"""The trace count on a small synthetic trace."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from akvr import trace
from akvr.spec import digest


def write_trace(path, rows):
    with gzip.open(path, 'wt') as f:
        f.write('\n'.join(json.dumps(r) for r in rows))


ROWS = [
    {'input_tokens_total': 90, 'output_tokens': 10, 'tools': [
        {'tool_wall_latency_ms': None}, {'tool_wall_latency_ms': 1000},
        {'tool_wall_latency_ms': 1001}, {}]},
    {'input_tokens_total': 32768, 'output_tokens': 1, 'tools': [
        {'tool_wall_latency_ms': None}, {'tool_wall_latency_ms': 2000}]},
    {'input_tokens_total': 5, 'output_tokens': 5, 'tools': []}]


class TraceCount(unittest.TestCase):
    def test_duration_boundary_and_unknown_durations_kept_overall(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'trace.gz'
            write_trace(path, ROWS)
            with self.assertRaisesRegex(ValueError, 'release file'):
                trace.count(path)
            data = trace.count(path, expected_sha=digest(path))
        self.assertEqual(data['rounds'], 3)
        result = trace.report(data)
        self.assertEqual((result['total_tools'], result['within_32768']), (6, 4))
        self.assertEqual(result['missing_duration_tools'], 3)
        subgroup = result['over_1s_subgroup']
        self.assertEqual((subgroup['total_tools'], subgroup['within_32768']), (2, 1))
        self.assertEqual(subgroup['excluded_missing_duration_within_32768'], 2)
        self.assertTrue(result['capacity_pending_keep_pool'])
        self.assertIsNone(result['capacity'])
        result = trace.report(data, {'r0': 100 * 57344, 'r1': 99 * 57344})
        self.assertEqual(result['capacity']['r1']['1']['exceeds'], 4)
        self.assertEqual(result['over_1s_capacity']['r0']['1']['exceeds'], 0)
        self.assertEqual(result['over_1s_capacity']['r1']['1']['exceeds'], 1)
        with self.assertRaises(ValueError):
            trace.report(data, {})
        with self.assertRaises(ValueError):
            trace.report(data, {'r0': 1.5})

    def test_denominators_count_tools_and_the_comparison_is_strict(self):
        data = {'histogram': {32768: 3, 32769: 4, 100: 2}, 'long_known_histogram': {100: 1},
                'unknown_duration_histogram': {32768: 1}}
        result = trace.report(data, {'round 0': 32768 * 57344})
        self.assertEqual(result['within_32768'], 5)
        self.assertEqual(result['capacity']['round 0']['1']['exceeds'], 0)
        self.assertEqual(result['capacity']['round 0']['2']['exceeds'], 3)
        self.assertEqual(result['capacity']['round 0']['2']['share'], 3 / 5)
        self.assertEqual(result['over_1s_capacity']['round 0']['2']['denominator'], 1)
        self.assertEqual(result['over_1s_capacity']['round 0']['2']['exceeds'], 0)

    def test_empty_long_subgroup_is_not_a_zero_share(self):
        data = {'histogram': {100: 1}, 'long_known_histogram': {}, 'unknown_duration_histogram': {100: 1}}
        cell = trace.report(data, {'r0': 1})['over_1s_capacity']['r0']['1']
        self.assertEqual(cell['denominator'], 0)
        self.assertIsNone(cell['share'])

    def test_invalid_rows_stop_the_count(self):
        for bad in ({'input_tokens_total': -1, 'output_tokens': 1, 'tools': [{}]},
                    {'input_tokens_total': 1, 'output_tokens': 1, 'tools': [{'tool_wall_latency_ms': -5}]},
                    {'input_tokens_total': 1, 'output_tokens': 1}):
            with tempfile.TemporaryDirectory() as tmp, self.subTest(bad=bad):
                path = Path(tmp) / 't.gz'
                write_trace(path, [bad])
                with self.assertRaises(ValueError):
                    trace.count(path, expected_sha=None)


if __name__ == '__main__':
    unittest.main()
