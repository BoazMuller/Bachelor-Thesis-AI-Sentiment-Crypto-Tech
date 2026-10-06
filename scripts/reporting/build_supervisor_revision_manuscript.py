"""Compatibility entry point for the reviewer-only manuscript rebuild.

The broader first-pass rewrite is archived in
thesis/revision_2026_10/before_reviewer_only_restart/.
"""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("build_reviewer_only_manuscript.py")), run_name="__main__")
