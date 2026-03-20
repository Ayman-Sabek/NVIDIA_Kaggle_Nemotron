"""Shared utilities for the Nemotron competition."""

import os
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
ADAPTERS_DIR = PROJECT_ROOT / "adapters"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"


def package_submission(adapter_dir: str | Path, output_name: str = "submission.zip") -> Path:
    """Package a LoRA adapter directory into a submission zip.
    
    Args:
        adapter_dir: Path to the adapter directory (must contain adapter_config.json)
        output_name: Name for the output zip file
    
    Returns:
        Path to the created zip file
    """
    adapter_dir = Path(adapter_dir)
    if not (adapter_dir / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"adapter_config.json not found in {adapter_dir}. "
            "This is required for a valid submission."
        )

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = SUBMISSIONS_DIR / output_name

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in adapter_dir.iterdir():
            if f.is_file():
                zf.write(f, f.name)

    print(f"Created submission: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")
    return output_path


def get_latest_adapter() -> Path | None:
    """Find the most recently modified adapter directory."""
    if not ADAPTERS_DIR.exists():
        return None
    adapters = [d for d in ADAPTERS_DIR.iterdir() if d.is_dir()]
    if not adapters:
        return None
    return max(adapters, key=lambda d: d.stat().st_mtime)
