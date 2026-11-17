"""Stage a training run on a free Kaggle GPU.

    uv run python scripts/kaggle_build.py            # writes data/kaggle/{arrays,kernel}
    kaggle datasets create -p data/kaggle/arrays     # first time (later: datasets version)
    cd data/kaggle/titikapi-train && kaggle kernels push -p . --accelerator NvidiaTeslaT4
    kaggle kernels output <user>/titikapi-train -p models/kaggle

The model inputs (data/cube/arrays, ~1.2 GB) go up once as a private dataset. The code travels
inside the kernel script, so code changes only need a new push.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CUBE = ROOT / "data" / "cube"
STAGE = ROOT / "data" / "kaggle"

KERNEL = '''"""TitikApi: run `titikapi` commands on Kaggle (scripts/kaggle_build.py)."""
import base64, glob, io, os, shutil, subprocess, sys, zipfile
from pathlib import Path

COMMANDS = {commands!r}  # `titikapi` CLI commands, run in order
CODE_ZIP = {code!r}

ROOT, OUT = Path("/tmp/titikapi"), Path("/kaggle/working")


def sh(*cmd, **kw):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True, **kw)


PIP = ("zarr>=3,<4", "xarray>=2025.1", "affine", "lightgbm")
sh(sys.executable, "-m", "pip", "install", "-q", *PIP)
zipfile.ZipFile(io.BytesIO(base64.b64decode(CODE_ZIP))).extractall(ROOT)

# model inputs -> local disk (faster random reads than /kaggle/input)
weather = glob.glob("/kaggle/input/**/weather.npy", recursive=True)
assert weather, "attach the titikapi-arrays dataset"
src = Path(weather[0]).parent
cube = ROOT / "data" / "cube"
shutil.copytree(src, cube / "arrays", ignore=shutil.ignore_patterns("*.zip", "*.zarr", "*.json*"))
shutil.copy(src / "stats.json", cube / "arrays")
# fire/static cubes were uploaded as cube_extra.zip, which Kaggle may or may not have unpacked
fire = glob.glob("/kaggle/input/**/fire.zarr", recursive=True)
if fire:
    base = Path(fire[0]).parent
    for name in ("fire.zarr", "static.zarr"):
        shutil.copytree(base / name, cube / name)
else:
    shutil.unpack_archive(glob.glob("/kaggle/input/**/cube_extra.zip", recursive=True)[0], cube)

# trained models from attached kernel outputs (e.g. to re-score them on new data)
for marker in glob.glob("/kaggle/input/**/models/*/calibration.json", recursive=True):
    model_dir = Path(marker).parent
    if not (ROOT / "models" / model_dir.name).exists():
        shutil.copytree(model_dir, ROOT / "models" / model_dir.name)
        print("attached model", model_dir.name, flush=True)

if shutil.which("nvidia-smi"):
    sh("nvidia-smi")
env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONUNBUFFERED="1")
failed = []
try:
    for cmd in COMMANDS:
        try:
            sh(sys.executable, "-m", "titikapi.cli", *cmd, cwd=ROOT, env=env)
        except subprocess.CalledProcessError as e:
            failed.append(" ".join(cmd))
            print("FAILED", e, flush=True)
finally:  # keep whatever was produced, even after a failure
    if (ROOT / "models").exists():
        shutil.copytree(ROOT / "models", OUT / "models", dirs_exist_ok=True)
    (OUT / "reports").mkdir(exist_ok=True)
    for f in (ROOT / "reports").glob("*.csv"):
        if not f.name.startswith("baselines_"):
            shutil.copy(f, OUT / "reports")
if failed:
    sys.exit(f"failed: {{failed}}")
'''


def kaggle_user() -> str:
    out = subprocess.run(["kaggle", "config", "view"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "username:" in line:
            return line.split("username:")[1].strip()
    raise SystemExit("kaggle CLI not configured (kaggle config view shows no username)")


def code_zip() -> str:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted((ROOT / "src").rglob("*.py")):
            z.write(f, f.relative_to(ROOT).as_posix())
        for f in sorted((ROOT / "reports").glob("baselines_*.csv")):
            z.write(f, f.relative_to(ROOT).as_posix())
    return base64.b64encode(buf.getvalue()).decode()


def stage_arrays(user: str) -> Path:
    """Hard links to the arrays (no copy) + the small cubes zipped + dataset metadata."""
    out = STAGE / "arrays"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    for f in (CUBE / "arrays").iterdir():
        os.link(f, out / f.name)
    with zipfile.ZipFile(out / "cube_extra.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("fire.zarr", "static.zarr"):
            for f in sorted((CUBE / name).rglob("*")):
                if f.is_file():
                    z.write(f, f.relative_to(CUBE).as_posix())
    meta = {"title": "titikapi-arrays", "id": f"{user}/titikapi-arrays"}
    meta["licenses"] = [{"name": "other"}]
    (out / "dataset-metadata.json").write_text(json.dumps(meta, indent=1))
    return out


def stage_kernel(
    user: str, slug: str, commands: list[list[str]], gpu: bool = True, sources=()
) -> Path:
    out = STAGE / slug
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    script = KERNEL.format(commands=commands, code=code_zip())
    (out / "titikapi_train.py").write_text(script, encoding="utf-8")
    meta = {
        "id": f"{user}/{slug}",
        "title": slug,
        "code_file": "titikapi_train.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": gpu,
        "enable_internet": True,
        "dataset_sources": [f"{user}/titikapi-arrays"],
        "competition_sources": [],
        "kernel_sources": [f"{user}/{k}" for k in sources],
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run",
        action="append",
        default=[],
        help='U-Net train + evaluate: "name [train args]", e.g. "unet_reg --epochs 12"',
    )
    p.add_argument("--cmd", action="append", default=[], help='any command, e.g. "train-lgbm"')
    p.add_argument("--kernel", default="titikapi-train", help="kernel slug")
    p.add_argument("--cpu", action="store_true", help="no GPU (e.g. for LightGBM)")
    p.add_argument("--source-kernel", action="append", default=[], help="attach its output")
    p.add_argument("--skip-arrays", action="store_true", help="only rebuild the kernel")
    args = p.parse_args()
    user = kaggle_user()
    if not args.skip_arrays:
        print("staged", stage_arrays(user))
    commands = []
    for r in args.run:
        name, *train_args = r.split()
        commands += [["train", "--name", name, *train_args], ["evaluate-model", "--name", name]]
    commands += [c.split() for c in args.cmd]
    commands = commands or [["train", "--name", "unet"], ["evaluate-model", "--name", "unet"]]
    stage = stage_kernel(user, args.kernel, commands, not args.cpu, args.source_kernel)
    print("staged", stage)


if __name__ == "__main__":
    main()
