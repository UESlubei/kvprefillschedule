"""Compatibility entry point for the complete V3 development scans.

Use full_pipeline.py for exclusive GPU stage scheduling; --plan-only is supported.
"""
from run_extended import main

if __name__ == "__main__":
    main()
