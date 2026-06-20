"""Unit tests for burnt_toast.metrics.MemoryTracker using a mocked psutil.

Uses process_name="fake_test_proc" throughout so these tests never touch
real psutil process state, and patches burnt_toast.metrics.psutil directly
(process_iter / Process) rather than monkeypatching the global psutil module.
"""

from __future__ import annotations

import time
import unittest
from collections import namedtuple
from unittest.mock import patch

import psutil

from burnt_toast.metrics import MemoryTracker

MemInfo = namedtuple("MemInfo", ["rss", "vms"])

MB = 1024 * 1024


class FakeProcess:
    """Stands in for psutil.Process(pid), backed by a live registry dict."""

    def __init__(self, pid: int, registry: dict) -> None:
        self.pid = pid
        self._registry = registry

    def name(self) -> str:
        if self.pid not in self._registry:
            raise psutil.NoSuchProcess(self.pid)
        return self._registry[self.pid]["name"]

    def memory_info(self) -> MemInfo:
        if self.pid not in self._registry:
            raise psutil.NoSuchProcess(self.pid)
        rec = self._registry[self.pid]
        return MemInfo(rss=rec["rss"], vms=rec["vms"])


class FakeIterEntry:
    def __init__(self, pid: int, name: str) -> None:
        self.info = {"pid": pid, "name": name}


class MemoryTrackerTestCase(unittest.TestCase):
    """Base class wiring a fake process registry into burnt_toast.metrics.psutil."""

    def setUp(self) -> None:
        self.registry: dict[int, dict] = {}

        def fake_process_iter(attrs=None):
            # Snapshot to avoid "dict changed size during iteration" if a
            # test mutates the registry from inside a callback.
            return [FakeIterEntry(pid, rec["name"]) for pid, rec in list(self.registry.items())]

        def fake_process(pid):
            return FakeProcess(pid, self.registry)

        patcher_iter = patch("burnt_toast.metrics.psutil.process_iter", side_effect=fake_process_iter)
        patcher_proc = patch("burnt_toast.metrics.psutil.Process", side_effect=fake_process)
        patcher_iter.start()
        patcher_proc.start()
        self.addCleanup(patcher_iter.stop)
        self.addCleanup(patcher_proc.stop)


class TestRescanAndSampling(MemoryTrackerTestCase):
    def test_rescan_finds_only_matching_process_name(self) -> None:
        self.registry[1] = {"name": "fake_test_proc", "rss": 100 * MB, "vms": 200 * MB}
        self.registry[2] = {"name": "other_proc", "rss": 999 * MB, "vms": 999 * MB}

        tracker = MemoryTracker(process_name="fake_test_proc")
        tracker._rescan()

        self.assertEqual(tracker._known_pids, {1})

    def test_sums_rss_across_multiple_matching_processes(self) -> None:
        # Models the Critic strategy: a main-model runner + a critic-model
        # runner loaded simultaneously -- both must be summed, not maxed.
        self.registry[1] = {"name": "fake_test_proc", "rss": 100 * MB, "vms": 50 * MB}
        self.registry[2] = {"name": "fake_test_proc", "rss": 300 * MB, "vms": 70 * MB}

        tracker = MemoryTracker(process_name="fake_test_proc")
        tracker._rescan()
        rss_mb, vms_mb = tracker._sample_and_update()

        self.assertAlmostEqual(rss_mb, 400.0, places=3)
        self.assertAlmostEqual(vms_mb, 120.0, places=3)
        self.assertAlmostEqual(tracker.peak_rss_mb, 400.0, places=3)
        self.assertTrue(tracker._ever_found)

    def test_ignores_non_matching_processes_entirely(self) -> None:
        self.registry[1] = {"name": "something_else", "rss": 999 * MB, "vms": 999 * MB}

        tracker = MemoryTracker(process_name="fake_test_proc")
        tracker._rescan()
        rss_mb, _ = tracker._sample_and_update()

        self.assertEqual(tracker._known_pids, set())
        self.assertEqual(rss_mb, 0.0)
        self.assertFalse(tracker._ever_found)

    def test_late_arriving_pid_is_picked_up_on_next_rescan(self) -> None:
        """The scenario that motivated periodic rediscovery: a model loads
        into a new runner subprocess AFTER tracking has already begun."""
        tracker = MemoryTracker(process_name="fake_test_proc")
        tracker._rescan()
        self.assertEqual(tracker._known_pids, set())

        # "Model loads" mid-run: a brand new matching process appears.
        self.registry[42] = {"name": "fake_test_proc", "rss": 500 * MB, "vms": 600 * MB}
        tracker._rescan()

        self.assertEqual(tracker._known_pids, {42})
        rss_mb, _ = tracker._sample_and_update()
        self.assertAlmostEqual(rss_mb, 500.0, places=3)

    def test_tolerates_process_disappearing_mid_run(self) -> None:
        self.registry[1] = {"name": "fake_test_proc", "rss": 200 * MB, "vms": 200 * MB}

        tracker = MemoryTracker(process_name="fake_test_proc")
        tracker._rescan()
        rss_mb, _ = tracker._sample_and_update()
        self.assertAlmostEqual(rss_mb, 200.0, places=3)

        del self.registry[1]
        rss_mb_after, _ = tracker._sample_and_update()

        self.assertEqual(rss_mb_after, 0.0)
        self.assertEqual(tracker._known_pids, set())
        # Peak must retain the earlier (higher) sample, not reset to 0.
        self.assertAlmostEqual(tracker.peak_rss_mb, 200.0, places=3)

    def test_tolerates_pid_reuse_by_unrelated_process(self) -> None:
        """A dead pid can be recycled by the OS for an unrelated process
        before the next rescan evicts it -- must not attribute that
        process's memory to the tracked Ollama process."""
        self.registry[7] = {"name": "fake_test_proc", "rss": 150 * MB, "vms": 150 * MB}

        tracker = MemoryTracker(process_name="fake_test_proc")
        tracker._rescan()
        tracker._sample_and_update()

        # PID 7 dies and the OS reuses it for an unrelated process.
        self.registry[7] = {"name": "completely_different", "rss": 9999 * MB, "vms": 9999 * MB}
        rss_mb, _ = tracker._sample_and_update()

        self.assertEqual(rss_mb, 0.0)
        self.assertEqual(tracker._known_pids, set())


class TestStartStopWarning(MemoryTrackerTestCase):
    def test_warns_once_when_never_found_anything(self) -> None:
        tracker = MemoryTracker(process_name="fake_test_proc", interval_seconds=0.01)
        tracker.start()
        time.sleep(0.05)
        with patch("burnt_toast.metrics.logger.warning") as mock_warning:
            tracker.stop()
        mock_warning.assert_called_once()
        self.assertEqual(tracker.peak_rss_mb, 0.0)

    def test_no_warning_when_found_during_run_even_if_gone_by_stop(self) -> None:
        self.registry[1] = {"name": "fake_test_proc", "rss": 100 * MB, "vms": 100 * MB}

        tracker = MemoryTracker(
            process_name="fake_test_proc",
            interval_seconds=0.01,
            rescan_interval_seconds=0.02,
        )
        tracker.start()
        time.sleep(0.05)
        del self.registry[1]  # process exits right before stop()
        time.sleep(0.05)

        with patch("burnt_toast.metrics.logger.warning") as mock_warning:
            tracker.stop()

        mock_warning.assert_not_called()
        self.assertAlmostEqual(tracker.peak_rss_mb, 100.0, places=1)


if __name__ == "__main__":
    unittest.main()
