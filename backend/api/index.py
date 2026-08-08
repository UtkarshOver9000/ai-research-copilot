import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from copilot.api.app import app  # noqa: E402

handler = app
