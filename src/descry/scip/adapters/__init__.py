"""SCIP language adapter registrations.

Importing this package eagerly loads every built-in adapter module, each of
which calls `register()` at import time so the global `ADAPTERS` registry is
populated. Downstream modules (`support`, `cache`, `parser`) iterate that
registry rather than hardcode per-language branches.
"""

from __future__ import annotations

# Side-effect imports: each module registers its adapter at load time.
from descry.scip.adapters import go, java, php, python, ruby, rust, typescript  # noqa: F401
