"""
Entry point for `python -m onyx`.

Delegates to onyx.cli.main().
"""

from __future__ import annotations

import sys

from onyx.cli import main

if __name__ == "__main__":
    sys.exit(main())
