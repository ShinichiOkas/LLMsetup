import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def jp_dir(tmp_path: Path) -> Path:
    """A path with Japanese characters and spaces, like real user folders."""
    d = tmp_path / "モデル 置き場" / "テスト 用"
    d.mkdir(parents=True)
    return d
