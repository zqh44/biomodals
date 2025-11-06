"""ABCFold2 Modal app client code.

Example usage:

    python -i clients/fold/abcfold2.py
    > run_abcfold2("path/to/example.yaml")
"""

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import modal

download_boltz_models = modal.Function.from_name("ABCFold2", "download_boltz_models")
download_chai_models = modal.Function.from_name("ABCFold2", "download_chai_models")
prepare_abcfold2 = modal.Function.from_name("ABCFold2", "prepare_abcfold2")
collect_abcfold2_boltz_data = modal.Function.from_name(
    "ABCFold2", "collect_abcfold2_boltz_data"
)
collect_abcfold2_chai_data = modal.Function.from_name(
    "ABCFold2", "collect_abcfold2_chai_data"
)


def run_abcfold2(
    input_yaml: str,
    run_name: str | None = None,
    download_models: bool = False,
    force_redownload: bool = False,
    run_boltz: bool = True,
    run_chai: bool = True,
) -> None:
    """Run ABCFold2 on modal and fetch results to $CWD.

    Args:
        input_yaml: Path to YAML design specification file
        run_name: Optional run name (defaults to timestamp-{input file hash})
        download_models: Whether to download model weights before running
        force_redownload: Whether to force re-download of model weights
        run_boltz: Whether to run Boltz inference
        run_chai: Whether to run Chai inference
    """
    if download_models:
        print("🧬 Checking Boltz inference dependencies...")
        download_boltz_models.remote(force=force_redownload)

        print("🧬 Checking Chai inference dependencies...")
        download_chai_models.remote(force=force_redownload)

    # Load input and find its hash
    yaml_path = Path(input_yaml).expanduser().resolve()
    yaml_str = yaml_path.read_bytes()

    run_id = hashlib.sha256(yaml_str).hexdigest()  # content-based id
    today: str = datetime.now(UTC).strftime("%Y%m%d%H%M")
    if run_name is None:
        run_name = run_id[:8]  # short id

    local_out_dir = Path.cwd() / f"{today}-{run_name}"
    if local_out_dir.exists():
        raise FileExistsError(f"Output directory already exists: {local_out_dir}")

    print(f"🧬 Starting ABCFold2 run {run_id}...")
    run_conf = prepare_abcfold2.remote(yaml_str=yaml_str, run_id=run_id)

    # Run Boltz for each seed
    if run_boltz:
        out_path = local_out_dir / f"boltz_{run_id}.tar.gz"
        print(f"🧬 Running Boltz and collecting results to {out_path}")
        boltz_data = collect_abcfold2_boltz_data.remote(run_conf=run_conf)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(boltz_data)

    # Run Chai for each seed
    if run_chai:
        out_path = local_out_dir / f"chai_{run_id}.tar.gz"
        print(f"🧬 Running Chai and collecting results to {out_path}")
        chai_data = collect_abcfold2_chai_data.remote(run_conf=run_conf)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(chai_data)

    print(f"🧬 ABCFold2 run complete! Results saved to {local_out_dir}")
