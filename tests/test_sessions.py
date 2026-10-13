"""Offline session construction with a small stand-in tokenizer (the real one is checked in the README)."""
import json
from pathlib import Path
import re
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from akvr import sessions
from akvr import spec
from akvr.spec import check_model, digest, ids_digest

SPECIAL = {'<|im_start|>': 900001, '<|im_end|>': 900002}


class CharTokenizer:
    """One id per character; the two chat markers are single special ids, as in Qwen2.5."""
    all_special_ids = list(SPECIAL.values())

    def encode(self, text, add_special_tokens=False):
        ids = []
        for part in re.split(r'(<\|im_start\|>|<\|im_end\|>)', text):
            ids.extend([SPECIAL[part]] if part in SPECIAL else [ord(c) for c in part])
        return ids

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = ''.join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return text + ('<|im_start|>assistant\n' if add_generation_prompt else '')


def source_files(count=12, size=4000):
    texts, files, offset = [], [], 0
    for i in range(count):
        words = ' '.join(f'w{i}_{j}' for j in range(size))
        text = f'# file {i}\n' + words[:size] + '\n' + ''.join(chr(0x4e00 + 300 * i + j) for j in range(300)) + '\n'
        files.append({'path': f'vllm/f{i:02d}.py', 'sha256': '', 'start': offset, 'end': offset + len(text)})
        texts.append(text)
        offset += len(text)
    return ''.join(texts), files


class Build(unittest.TestCase):
    def test_lengths_distinct_first_blocks_and_no_special_ids_in_content(self):
        tok = CharTokenizer()
        text, files = source_files()
        data = sessions.build(tok, text, files, rounds=1, levels=(2048,))
        sessions.validate(data, rounds=1, levels=(2048,))
        self.assertEqual(len(data['sessions']), 5)
        for s in data['sessions']:
            self.assertEqual(len(s['prompt_ids']), 2048 - 768)
            self.assertEqual(len(s['closing_ids']), 256)
            self.assertEqual(s['prompt_ids'], tok.encode(s['prompt_text']))
            self.assertTrue(s['prompt_text'].endswith('<|im_start|>assistant\n'))
            self.assertTrue(s['closing_text'].startswith('<|im_end|>\n<|im_start|>user\n'))
            self.assertTrue(s['closing_text'].endswith('<|im_start|>assistant\n'))
            self.assertIn(s['tool_text'], s['closing_text'])
        self.assertEqual(len({tuple(s['prompt_ids'][:16]) for s in data['sessions']}), 5)
        self.assertEqual([s['warmup'] for s in data['sessions']], [True, False, False, False, False])
        self.assertTrue(set(SPECIAL.values()).isdisjoint(data['filler_vocabulary']))
        # Deterministic: the same inputs give the same ids.
        again = sessions.build(tok, text, files, rounds=1, levels=(2048,))
        self.assertEqual(ids_digest(again['sessions']), ids_digest(data['sessions']))

    def test_validate_rejects_wrong_lengths_duplicates_and_gaps(self):
        tok = CharTokenizer()
        text, files = source_files()
        data = sessions.build(tok, text, files, rounds=1, levels=(2048,))
        bad = json.loads(json.dumps(data))
        bad['sessions'][1]['prompt_ids'].append(5)
        with self.assertRaisesRegex(ValueError, 'lengths'):
            sessions.validate(bad, rounds=1, levels=(2048,))
        bad = json.loads(json.dumps(data))
        bad['sessions'][2]['prompt_ids'][:16] = bad['sessions'][1]['prompt_ids'][:16]
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            sessions.validate(bad, rounds=1, levels=(2048,))
        bad = json.loads(json.dumps(data))
        bad['sessions'].pop()
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            sessions.validate(bad, rounds=1, levels=(2048,))
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            sessions.validate(data)  # the full set has five rounds and three levels
        bad = json.loads(json.dumps(data))
        bad['model'] = 'other/model'
        with self.assertRaisesRegex(ValueError, 'another model'):
            sessions.validate(bad, rounds=1, levels=(2048,))

    def test_trim_finds_exact_token_count_or_refuses(self):
        tok = CharTokenizer()
        self.assertEqual(sessions.trim('abcdefghij', 4, lambda s: len(tok.encode(s))), 'abcd')
        with self.assertRaises(ValueError):
            sessions.trim('abc', 10, lambda s: len(tok.encode(s)))


class SourceTree(unittest.TestCase):
    def test_reads_committed_python_files_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'vllm/sub').mkdir(parents=True)
            (root / 'vllm/a.py').write_text('print(1)\n')
            (root / 'vllm/sub/b.py').write_text('x = 2\n')
            (root / 'vllm/notes.txt').write_text('ignored\n')
            git = ['git', '-C', tmp, '-c', 'user.name=t', '-c', 'user.email=t@example.com']
            subprocess.run(git[:3] + ['init', '-q'], check=True)
            subprocess.run(git + ['add', '.'], check=True)
            subprocess.run(git + ['commit', '-qm', 'x'], check=True)
            head = subprocess.check_output(git[:3] + ['rev-parse', 'HEAD'], text=True).strip()
            (root / 'vllm/a.py').write_text('edited in the working tree\n')
            with patch.object(sessions, 'COMMIT', head):
                text, files = sessions.source_material(root)
            self.assertEqual(text, 'print(1)\nx = 2\n')
            self.assertEqual([f['path'] for f in files], ['vllm/a.py', 'vllm/sub/b.py'])
            self.assertEqual((files[1]['start'], files[1]['end']), (9, 15))
            with self.assertRaisesRegex(ValueError, 'is at .*, not '):
                sessions.source_material(root)


class Model(unittest.TestCase):
    CFG = {'hidden_size': 3584, 'num_attention_heads': 28, 'num_hidden_layers': 28,
           'num_key_value_heads': 4, 'max_position_embeddings': 32768, 'torch_dtype': 'bfloat16'}

    def snapshot(self, tmp, cfg=None):
        snap = Path(tmp) / 'snapshots/a09a35458c702b33eeacc393d103063234e8bc28'
        snap.mkdir(parents=True, exist_ok=True)
        (snap / 'config.json').write_text(json.dumps(cfg or self.CFG))
        (snap / 'generation_config.json').write_text('{"top_k": 20}')
        (snap / 'tokenizer.json').write_text('{}')
        (snap / 'tokenizer_config.json').write_text('{}')
        return snap

    def reference_of(self, snap):
        return {name: digest(snap / name) for name in spec.REFERENCE_MODEL_FILES}

    def test_reference_snapshot_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            snap = self.snapshot(tmp)
            with patch.dict(spec.REFERENCE_MODEL_FILES, self.reference_of(snap)):
                info = check_model(snap)
            self.assertTrue(info['reference'])
            self.assertEqual(info['revision'], 'a09a35458c702b33eeacc393d103063234e8bc28')

    def test_other_copy_is_refused_unless_allowed_and_then_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            snap = self.snapshot(tmp)
            with patch.dict(spec.REFERENCE_MODEL_FILES, self.reference_of(snap)):
                (snap / 'generation_config.json').write_text('{"top_k": 50}')
                with self.assertRaisesRegex(ValueError, 'generation_config.json.*--allow-other-model'):
                    check_model(snap)
                info = check_model(snap, allow_other=True)
                self.assertEqual((info['reference'], info['different_files'], info['allow_other_model']),
                                 (False, ['generation_config.json'], True))
                (snap / 'generation_config.json').unlink()
                with self.assertRaisesRegex(ValueError, 'lacks generation_config.json'):
                    check_model(snap, allow_other=True)

    def test_the_real_reference_hashes_reject_a_stand_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'Not the reference model snapshot'):
                check_model(self.snapshot(tmp))

    def test_model_geometry_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            for key, value in (('num_key_value_heads', 8), ('max_position_embeddings', 131072),
                               ('torch_dtype', 'float16')):
                snap = self.snapshot(tmp, self.CFG | {key: value})
                with self.subTest(key=key), self.assertRaises(ValueError):
                    check_model(snap, allow_other=True)


if __name__ == '__main__':
    unittest.main()
