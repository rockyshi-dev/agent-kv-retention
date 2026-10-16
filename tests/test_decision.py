"""The checks per arm and level and the decision row, on synthetic sessions."""
import copy
import unittest

from akvr import gates
from akvr.spec import ARMS, LEVELS, host_min, reusable

from .helpers import bundle


class Arithmetic(unittest.TestCase):
    def test_reusable_and_host_minimum(self):
        # P = N - 768; C = floor((P + 511) / 16) * 16.
        self.assertEqual([reusable(n) for n in LEVELS], [1776, 7920, 28400])
        self.assertEqual(host_min('NATIVE', 28672), 27904 - 16)
        self.assertEqual(host_min('NATIVE-ALL', 28672), 28400 - 16)
        self.assertEqual(host_min('LAZY', 2048), 1776 - 16)


class Turn1Agreement(unittest.TestCase):
    def test_first_difference(self):
        self.assertIsNone(gates.first_difference([1, 2, 3], [1, 2, 3]))
        self.assertEqual(gates.first_difference([1, 2, 3], [1, 9, 3]), 1)
        self.assertEqual(gates.first_difference([5], [1]), 0)
        self.assertEqual(gates.first_difference([1, 2], [1, 2, 3]), 2)

    def test_base_is_the_first_full_turn_in_the_sessions_order_and_warmups_are_not_counted(self):
        data = bundle(.7)
        rows = data['rounds'][0]['sessions']
        rows.insert(0, {'id': 'r0-n2048-s0', 'arm': 'KEEP', 'N': 2048, 'round': 0, 'index': 0, 'warmup': True,
                        'turn1': {'token_ids': [1] * 512}})
        for s in rows:
            if s['arm'] == 'KEEP' and s['N'] == 2048 and s['index'] == 1:
                s['turn1']['token_ids'] = [1] * 100  # too short: neither a base nor compared
            if s['arm'] == 'DROP' and s['N'] == 2048:
                s['turn1']['token_ids'] = [17] * 5 + [1] * 507
        item = gates.turn1_agreement(data['rounds'])['2048']['0']
        self.assertEqual(item['base_arms'], ['DROP', 'KEEP'])  # session 1's base is DROP, the others' KEEP
        self.assertEqual((item['sessions'], item['compared'], item['agree']), (20, 15, 9))
        self.assertEqual([(d['arm'], d['index'], d['base_arm'], d['first_diff_from_base']) for d in item['differing']],
                         [('DROP', 2, 'KEEP', 5), ('DROP', 3, 'KEEP', 5), ('DROP', 4, 'KEEP', 5),
                          ('NATIVE', 1, 'DROP', 5), ('NATIVE-ALL', 1, 'DROP', 5), ('LAZY', 1, 'DROP', 5)])
        self.assertEqual(gates.compute(data)['row'], 'RESTORE-WINS')


class Decisions(unittest.TestCase):
    def test_winning_row_requires_every_round_and_both_levels(self):
        data = bundle(.8)
        self.assertEqual(gates.compute(data)['row'], 'RESTORE-WINS')
        for s in data['rounds'][2]['sessions']:
            if s['arm'] in ARMS[2:] and s['N'] == 8192:
                s['resume']['ttft_ms'] = 81
        self.assertEqual(gates.compute(data)['row'], 'INCONCLUSIVE')

    def test_one_winning_arm_is_enough(self):
        data = bundle(.9)
        for r in data['rounds']:
            for s in r['sessions']:
                if s['arm'] == 'NATIVE-ALL':
                    s['resume']['ttft_ms'] = 50
        result = gates.compute(data)
        self.assertEqual(result['row'], 'RESTORE-WINS')
        self.assertEqual(len(result['ratios_by_round']['NATIVE/28672']), 5)

    def test_no_win_requires_every_h_passing_arm_to_have_ratios(self):
        data = bundle(.9)
        for r in data['rounds']:
            for s in r['sessions']:
                if s['arm'] == 'LAZY':
                    s['resume']['ttft_ms'] = 50
        lazy = [s for r in data['rounds'] for s in r['sessions'] if s['arm'] == 'LAZY' and s['N'] == 28672]
        for s in lazy[:2]:
            s['turn1']['token_ids'] = []
        result = gates.compute(data)
        self.assertTrue(result['cells']['LAZY/28672']['H'])
        self.assertFalse(result['cells']['LAZY/28672']['valid'])
        self.assertEqual(result['row'], 'INCONCLUSIVE')

    def test_one_l_drop_plus_one_e_drop_invalidates(self):
        data = bundle()
        rows = [s for r in data['rounds'] for s in r['sessions'] if s['arm'] == 'DROP' and s['N'] == 28672]
        rows[0]['turn1']['token_ids'] = []
        rows[5]['resume']['delta']['prefix_cache_hits'] = 4096
        cell = gates.compute(data)['cells']['DROP/28672']
        self.assertEqual((cell['L_dropped'], cell['E_dropped'], cell['valid']), (1, 1, False))

    def test_load_bytes_contradicting_hh_is_flagged(self):
        data = bundle()
        for r in data['rounds']:
            for s in r['sessions']:
                if s['arm'] == 'NATIVE':
                    s['resume']['delta']['kv_offload_load_bytes'] = 0
        result = gates.compute(data)
        self.assertEqual(result['cells']['NATIVE/28672']['load_mismatches'], 20)
        self.assertTrue(any('host-load bytes' in e for e in result['examine']))
        self.assertEqual(gates.compute(bundle())['cells']['NATIVE/28672']['load_mismatches'], 0)

    def test_round_with_different_pools_invalidates_every_arm(self):
        data = bundle()
        self.assertEqual(gates.compute(data)['row'], 'RESTORE-WINS')
        data['rounds'][2]['starts']['LAZY']['pool_tokens'] = 84864
        result = gates.compute(data)
        self.assertEqual(result['pool_mismatch_rounds'], [2])
        self.assertTrue(all(not c['valid'] for c in result['cells'].values()))
        self.assertNotEqual(result['row'], 'RESTORE-WINS')

    def test_no_win_cannot_be_vacuous(self):
        data = bundle(.91)
        self.assertEqual(gates.compute(data)['row'], 'NO-WIN')
        for r in data['rounds']:
            for s in r['sessions']:
                if s['arm'] in ARMS[2:]:
                    s['resume']['delta']['external_prefix_cache_hits'] = 0
        out = gates.compute(data)
        self.assertEqual(out['row'], 'INCONCLUSIVE')
        self.assertTrue(out['examine'])

    def test_missing_round_and_duplicates_cannot_pass(self):
        data = bundle()
        data['rounds'].pop()
        self.assertEqual(gates.compute(data)['row'], 'INCONCLUSIVE')
        data = bundle()
        data['rounds'][0]['sessions'].append(data['rounds'][0]['sessions'][0])
        with self.assertRaises(ValueError):
            gates.compute(data)

    def test_keep_failure_dominates_winning_ratios(self):
        data = bundle()
        for s in data['rounds'][0]['sessions']:
            if s['arm'] == 'KEEP' and s['N'] == 28672:
                s['resume']['delta']['prefix_cache_hits'] = 0
        self.assertEqual(gates.compute(data)['row'], 'NOT-MEASURABLE')

    def test_keep_failure_at_a_shorter_level_invalidates_only_that_level(self):
        data = bundle()
        for s in data['rounds'][1]['sessions']:
            if s['arm'] == 'KEEP' and s['N'] == 2048:
                s['resume']['delta']['prefix_cache_hits'] = 0
        result = gates.compute(data)
        self.assertTrue(all(not result['cells'][f'{a}/2048']['valid'] for a in ARMS))
        self.assertTrue(result['cells']['NATIVE/8192']['valid'])
        self.assertEqual(result['row'], 'RESTORE-WINS')

    def test_length_drop_limit_is_over_twenty_not_per_round(self):
        data = bundle()
        bad = [s for r in data['rounds'] for s in r['sessions'] if s['arm'] == 'LAZY' and s['N'] == 2048]
        bad[0]['turn1']['token_ids'] = []
        self.assertTrue(gates.compute(data)['cells']['LAZY/2048']['valid'])
        bad[4]['resume']['usage']['prompt_tokens'] = 2047
        self.assertFalse(gates.compute(data)['cells']['LAZY/2048']['valid'])

    def test_keep_length_failures_do_not_invalidate_other_arms(self):
        data = bundle()
        bad = [s for r in data['rounds'] for s in r['sessions'] if s['arm'] == 'KEEP' and s['N'] == 28672]
        for s in bad[:2]:
            s['turn1']['token_ids'] = []
        result = gates.compute(data)
        self.assertFalse(result['cells']['KEEP/28672']['valid'])
        self.assertTrue(result['cells']['KEEP/28672']['K'])
        self.assertTrue(result['cells']['NATIVE/28672']['valid'])
        self.assertEqual(result['row'], 'RESTORE-WINS')

    def test_no_usable_keep_session_is_k_failure(self):
        for cause in ('missing', 'length', 'start'):
            with self.subTest(cause=cause):
                data = bundle()
                for r in data['rounds']:
                    if cause == 'missing':
                        r['sessions'] = [s for s in r['sessions'] if s['arm'] != 'KEEP']
                    elif cause == 'start':
                        r['starts']['KEEP']['pass'] = False
                    else:
                        for s in r['sessions']:
                            if s['arm'] == 'KEEP':
                                s['turn1']['token_ids'] = []
                result = gates.compute(data)
                self.assertFalse(result['cells']['KEEP/28672']['K'])
                self.assertFalse(result['cells']['DROP/28672']['valid'])
                self.assertEqual(result['row'], 'NOT-MEASURABLE')

    def test_host_failure_not_hidden_by_median(self):
        data = bundle()
        bad = next(s for s in data['rounds'][3]['sessions'] if s['arm'] == 'NATIVE-ALL' and s['N'] == 8192)
        bad['resume']['delta']['external_prefix_cache_hits'] = 10
        result = gates.compute(data)
        self.assertNotIn('NATIVE-ALL/8192', result['ratios_by_round'])
        self.assertEqual(result['cells']['NATIVE-ALL/8192']['host_status'], 'host copy incomplete')

    def test_failed_quiet_attempt_is_excluded(self):
        data = bundle()
        bad = copy.deepcopy(data['rounds'][0])
        bad['selected'] = False
        bad['quiet'] = {'pass': False}
        for s in bad['sessions']:
            s['resume']['ttft_ms'] = 999999
        data['rounds'].insert(0, bad)
        self.assertEqual(gates.compute(data)['row'], 'RESTORE-WINS')
        data['rounds'][0]['quiet_exhausted'] = True
        self.assertEqual(gates.compute(data)['row'], 'NOT-MEASURABLE')

    def test_native_measured_session_without_load_counter_raises(self):
        data = bundle()
        row = next(s for s in data['rounds'][0]['sessions'] if s['arm'] == 'NATIVE')
        row['resume']['delta']['kv_offload_load_bytes'] = None
        with self.assertRaisesRegex(ValueError, 'kv_offload_load_bytes'):
            gates.session_gates(row)
        row['warmup'] = True
        self.assertIsNone(gates.session_gates(row)['load_tokens_crosscheck'])

    def test_differences_and_token_counts(self):
        data = bundle()
        for r in data['rounds']:
            for s in r['sessions']:
                if s['arm'] == 'NATIVE':
                    s['resume']['ttft_ms'] = 90
        result = gates.compute(data)
        item = result['differences_by_round']['28672']['0']
        self.assertAlmostEqual(item['default_cost_ttft_ms'], 20)
        self.assertEqual(item['default_cost_recomputed_tokens'], 768 - 272)
        self.assertAlmostEqual(item['eager_minus_lazy_ttft_ms'], 0)
        self.assertEqual(result['trace_pool_bytes_by_round']['0'], 57344 * 80000)


if __name__ == '__main__':
    unittest.main()
