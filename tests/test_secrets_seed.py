"""Unit tests for burnt_toast.secrets.generate_run_secret determinism.

The cross-process test below is the only one that can actually catch the
original bug: Python's builtin hash() is randomized per process via
PYTHONHASHSEED, but that seed is fixed for the whole lifetime of a single
process -- so a same-process test would pass even on the old buggy code
(hash(key) would be internally consistent throughout the test run). Only
spawning separate subprocesses proves the secret code is stable across
process boundaries, as the docstring claims.
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

from burnt_toast.secrets import generate_run_secret

REPO_ROOT = Path(__file__).resolve().parent.parent

_SNIPPET = (
    "from burnt_toast.secrets import generate_run_secret; "
    "s = generate_run_secret(run_index=0, model='qwen2.5:1.5b', "
    "context_size_tokens=4000, needle_position='middle', "
    "strategy='No-Guard', experiment_mode='needle'); "
    "print(s.code)"
)


class TestSecretDeterminismSameProcess(unittest.TestCase):
    def test_same_config_same_code_in_process(self) -> None:
        """Basic sanity check only -- does NOT prove the cross-process
        guarantee, since hash() would also pass this within one process."""
        kwargs = dict(
            run_index=0,
            model="qwen2.5:1.5b",
            context_size_tokens=4000,
            needle_position="middle",
            strategy="No-Guard",
            experiment_mode="needle",
        )
        first = generate_run_secret(**kwargs)
        second = generate_run_secret(**kwargs)
        self.assertEqual(first.code, second.code)
        self.assertEqual(first.phrase, second.phrase)

    def test_different_run_index_different_code(self) -> None:
        kwargs = dict(
            model="qwen2.5:1.5b",
            context_size_tokens=4000,
            needle_position="middle",
            strategy="No-Guard",
            experiment_mode="needle",
        )
        a = generate_run_secret(run_index=0, **kwargs)
        b = generate_run_secret(run_index=1, **kwargs)
        self.assertNotEqual(a.code, b.code)


class TestSecretDeterminismCrossProcess(unittest.TestCase):
    def test_same_config_same_code_across_processes(self) -> None:
        # Strip PYTHONHASHSEED (rather than just inheriting it) so each child
        # gets an independently-random per-process hash seed -- if the test
        # runner's own environment happened to pin PYTHONHASHSEED (e.g. under
        # some CI setups), inheriting it would make this test pass even on
        # the old buggy hash()-based implementation, a false negative.
        env = {k: v for k, v in os.environ.items() if k != "PYTHONHASHSEED"}

        out1 = subprocess.run(
            [sys.executable, "-c", _SNIPPET],
            capture_output=True, text=True, env=env, cwd=REPO_ROOT,
        )
        out2 = subprocess.run(
            [sys.executable, "-c", _SNIPPET],
            capture_output=True, text=True, env=env, cwd=REPO_ROOT,
        )

        self.assertEqual(out1.returncode, 0, out1.stderr)
        self.assertEqual(out2.returncode, 0, out2.stderr)
        self.assertEqual(out1.stdout.strip(), out2.stdout.strip())


if __name__ == "__main__":
    unittest.main()
