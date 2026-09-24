"""Make the test suite independent of how (or whether) the package was installed.

On the first GPU run ``pytest tests/`` failed at collection with

    ModuleNotFoundError: No module named 'cvdyn2dgs.metrics.clinical'

while the CLI imported the very same module without trouble - ``cvdyn2dgs demo`` passed,
and it pulls in ``metrics.clinical`` transitively. So the module was importable; what
differed was *which* ``cvdyn2dgs`` pytest resolved. An editable install whose finder was
written at a different time, a second interpreter's ``site-packages``, or a stale
``__editable__`` mapping all produce exactly this: the top-level package resolves, a
submodule does not.

Putting the repository root first on ``sys.path`` removes the ambiguity - the tests then
exercise the checkout they live in, which is what a test run is supposed to verify.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
