#!/usr/bin/env python3
"""
Convenience launcher for the Burnt Toast benchmark.

Usage:
    python run_benchmark.py                  # full matrix
    python run_benchmark.py --quick          # smoke test
    python run_benchmark.py --dry-run        # preview runs
    python run_benchmark.py --hardware-env Raspberry_Pi_5
"""

from burnt_toast.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
