"""Make `baseline/` importable from the test suite without shadowing `momentGW`.

`baseline/` is deliberately not part of the distribution (`packages = ["momentGW"]` in
`pyproject.toml`), so a test that imports it needs the repository root on the path -
otherwise bare `pytest` fails collection for the whole suite and only `python -m pytest`
works.

The root is **appended**, not prepended, and this is the point. `pythonpath = ["."]` under
`[tool.pytest.ini_options]` would put it ahead of site-packages, so `import momentGW` would
resolve to the working tree even when CI has just built and installed a wheel - the test job
would exercise the checkout rather than the artefact, and a packaging regression would ship
green. Appending leaves an installed `momentGW` winning and only fills the gap `baseline/`
leaves, which is the whole reason this file exists.
"""

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.append(ROOT)
