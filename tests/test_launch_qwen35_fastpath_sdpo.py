import importlib.util
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_PATH = PROJECT_ROOT / "tools" / "launch_qwen35_fastpath_sdpo.py"
SPEC = importlib.util.spec_from_file_location("launch_qwen35_fastpath_sdpo", LAUNCHER_PATH)
assert SPEC is not None
launcher = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(launcher)


@pytest.mark.parametrize(
    "image",
    [
        "zetermordio/anchor-qwen35-fastpath:torch2.10-cu128-fla0.5.0",
        "hf.co/spaces/ZeterMordio/anchor-qwen35-fastpath",
        "docker.io/zetermordio/anchor-qwen35-fastpath:torch2.10-cu128-fla0.5.0",
    ],
)
def test_validate_remote_image_ref_accepts_pullable_refs(image):
    launcher.validate_remote_image_ref(image)


@pytest.mark.parametrize(
    "image",
    [
        "anchor-qwen35-fastpath:torch2.10-cu128-fla0.5.0",
        "localhost/anchor-qwen35-fastpath:torch2.10-cu128-fla0.5.0",
        "127.0.0.1/anchor-qwen35-fastpath:torch2.10-cu128-fla0.5.0",
    ],
)
def test_validate_remote_image_ref_rejects_local_only_refs(image):
    with pytest.raises(SystemExit):
        launcher.validate_remote_image_ref(image)
