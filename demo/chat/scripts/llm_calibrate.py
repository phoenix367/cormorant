#!/usr/bin/env python3
"""
llm_calibrate.py — the study stage of the Llama decoders from pinned inputs:
checkpoint and texts fetched and verified, the power-of-two exponents
calibrated (llm_study.py formats) and checked against the recorded hash,
provenance written next to the result (doc/plans/CHAT_PLAN.md §10, §20).

llm_models.json (tracked) pins every input and the expected output:
  models.<name>  Hugging Face repo + revision, the SHA-256 of each checkpoint
                 file, the calibration policy and the SHA-256 of its formats
                 JSON (what generate_llm_project.py builds the library from),
                 optionally the shipped policy's study metrics;
  texts          the WikiText-2 held-out / calibration texts (SHA-256).

  fetch MODEL      checkpoint at the pinned revision -> assets/<model>/, texts
                   (llm_study.py fetch, or a verified copy from another model's
                   assets); every SHA-256 checked, matching files kept
  calibrate MODEL  llm_study.py formats into a scratch file; installed as
                   assets/study[/<model>]/formats_<policy>.json with a provenance
                   JSON (input hashes, code version, environment) when its SHA-256
                   equals the manifest's; otherwise left as formats_<policy>.new.json
                   (--force installs it; --record installs it and stores its hash:
                   a new model or an intended change)
  check MODEL      the same without installing: compare with the manifest and the
                   installed file; on a difference, list the exponents / sink
                   values that differ
  study MODEL      llm_study.py study for bf16 and the shipped policies ->
                   <study dir>/shipped/, the shipped policy's metrics compared
                   with the manifest (--record stores them)
  validate MODEL   llm_study.py validate (numpy float64 against torch float32)
  all MODEL        fetch + calibrate (+ validate / study with --validate / --study)
  add MODEL --repo R [--revision REV]   (maintainer) pin a new checkpoint:
                   download it, record the revision's commit and the file hashes

The study steps run in the study environment (numpy + tokenizers; validate
also torch + transformers): --study-python or STUDY_PYTHON, default
<repo>/.venv-export/bin/python — create it from requirements-study.txt.
The calibration is deterministic on one machine; numpy's OpenBLAS picks
CPU-specific kernels, so another machine may round a sink value
differently — `check` names such differences.

usage: python3 demo/chat/scripts/llm_calibrate.py all smollm2-360m-instruct
       python3 demo/chat/scripts/llm_calibrate.py check smollm2-135m-instruct
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent
CHAT = HERE.parent
REPO = CHAT.parent.parent
ASSETS = CHAT / "assets"
MANIFEST = HERE / "llm_models.json"
STUDY = HERE / "llm_study.py"
DEFAULT_MODEL = "smollm2-135m-instruct"          # llm_study.DEFAULT_MODEL: study files in assets/study
HF = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
SHIPPED_POLICIES = "bf16,pow2+sink+p12,pow2+sink+p12+mix"
SHIPPED = "pow2+sink+p12+mix"
CHECKPOINT_FILES = ("config.json", "generation_config.json", "model.safetensors", "tokenizer.json",
                    "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt")


class CalibrationError(RuntimeError):
    pass


# ── files and hashes ─────────────────────────────────────────────────────────

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest(path: Path = MANIFEST) -> dict:
    return json.loads(Path(path).read_text())


def save_manifest(m: dict, path: Path = MANIFEST) -> None:
    with open(path, "w") as f:
        json.dump(m, f, indent=2)
        f.write("\n")


def model_spec(m: dict, model: str) -> dict:
    try:
        return m["models"][model]
    except KeyError:
        raise CalibrationError(f"{model}: not in the manifest (known: {', '.join(sorted(m['models']))};"
                               f" pin a new one with `add`)") from None


def study_dir(root: Path, model: str) -> Path:
    """llm_study.study_dir for assets under ``root``."""
    return root / "study" if model == DEFAULT_MODEL else root / "study" / model


def download(url: str, dest: Path, want: Optional[str]) -> str:
    """Stream url into dest (via a .part file) and return its SHA-256; with
    ``want`` a mismatch removes the file and raises."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    h = hashlib.sha256()
    req = urllib.request.Request(url, headers={"User-Agent": "axi_demo-llm_calibrate/1"})
    with urllib.request.urlopen(req, timeout=120) as r, open(part, "wb") as f:
        for block in iter(lambda: r.read(1 << 20), b""):
            h.update(block)
            f.write(block)
    got = h.hexdigest()
    if want and got != want:
        part.unlink()
        raise CalibrationError(f"{dest.name}: SHA-256 {got} != pinned {want} ({url})")
    part.replace(dest)
    return got


def verify_inputs(m: dict, model: str, root: Path) -> Dict[str, str]:
    """{file: sha256} of the model's checkpoint files and the texts, checked
    against the manifest (raises naming what is missing or different)."""
    spec = model_spec(m, model)
    d = root / model
    want = dict(spec["files"])
    want.update({n: t["sha256"] for n, t in m["texts"].items()})
    got, bad = {}, []
    for name, h in want.items():
        p = d / name
        if not p.exists():
            bad.append(f"{name} missing")
            continue
        got[name] = sha256_file(p)
        if got[name] != h:
            bad.append(f"{name} differs")
    if bad:
        raise CalibrationError(f"{model} inputs in {d}: {'; '.join(bad)} — run `fetch {model}`")
    return got


# ── the study environment ────────────────────────────────────────────────────

def study_python(arg: Optional[str]) -> str:
    py = arg or os.environ.get("STUDY_PYTHON") or str(REPO / ".venv-export" / "bin" / "python")
    if not Path(py).exists():
        raise CalibrationError(f"study python {py} not found — create the environment from "
                               f"{HERE / 'requirements-study.txt'} or pass --study-python")
    return py


def run_study(py: str, args: List[str], log: bool = True) -> None:
    cmd = [py, str(STUDY)] + args
    if log:
        print("  $ " + " ".join(cmd), flush=True)
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        raise CalibrationError(f"llm_study.py {args[0]} failed (rc {rc})")


def environment(py: str) -> dict:
    """Versions that determine the calibration's floating-point results."""
    code = (
        "import json, platform\n"
        "d = {'python': platform.python_version()}\n"
        "try:\n"
        "    import numpy as np; d['numpy'] = np.__version__\n"
        "    import tokenizers; d['tokenizers'] = tokenizers.__version__\n"
        "except ImportError: pass\n"
        "try:\n"
        "    c = np.show_config(mode='dicts')['Build Dependencies']['blas']\n"
        "    d['blas'] = f\"{c.get('name')} {c.get('version')}\"\n"
        "except Exception: pass\n"
        "print(json.dumps(d))\n")
    out = subprocess.run([py, "-c", code], capture_output=True, text=True, check=True).stdout
    env = json.loads(out)
    env["platform"] = platform.platform()
    try:
        cpu = next(ln.split(":", 1)[1].strip() for ln in Path("/proc/cpuinfo").read_text().splitlines()
                   if ln.startswith("model name"))
    except (OSError, StopIteration):
        cpu = platform.processor()
    env["cpu"] = cpu
    return env


def code_version() -> dict:
    def git(*a):
        return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True,
                               text=True).stdout.strip()
    dirty = git("status", "--porcelain", "--", str(STUDY.relative_to(REPO)))
    return {"git_commit": git("rev-parse", "HEAD"), "llm_study_modified": bool(dirty),
            "llm_study_sha256": sha256_file(STUDY)}


# ── formats comparison ───────────────────────────────────────────────────────

def diff_formats(a: Path, b: Path, limit: int = 12) -> List[str]:
    """Human-readable differences between two formats JSONs."""
    x, y = json.loads(Path(a).read_text()), json.loads(Path(b).read_text())
    out = []
    for key in ("policy", "spec", "margin", "sink_token"):
        if x.get(key) != y.get(key):
            out.append(f"{key}: {x.get(key)} != {y.get(key)}")
    ex, ey = x.get("exponents", {}), y.get("exponents", {})
    for k in sorted(set(ex) | set(ey)):
        if ex.get(k) != ey.get(k):
            u, v = ex.get(k), ey.get(k)
            if isinstance(u, list) and isinstance(v, list) and len(u) == len(v):
                n = sum(p != q for p, q in zip(u, v))
                out.append(f"exponents[{k}]: {n} of {len(u)} channels differ")
            else:
                out.append(f"exponents[{k}]: {u} != {v}")
    for key in ("sink_k_raw", "sink_v_raw"):
        if x.get(key) != y.get(key):
            flat = lambda z: [v for layer in (z or []) for row in layer for v in (row if isinstance(row, list) else [row])]
            fx, fy = flat(x.get(key)), flat(y.get(key))
            n = sum(p != q for p, q in zip(fx, fy)) + abs(len(fx) - len(fy))
            mx = max((abs(p - q) for p, q in zip(fx, fy)), default=0)
            out.append(f"{key}: {n} of {max(len(fx), len(fy))} values differ (max |diff| {mx})")
    return out[:limit] + ([f"... {len(out) - limit} more"] if len(out) > limit else [])


# ── commands ─────────────────────────────────────────────────────────────────

def cmd_fetch(m: dict, model: str, root: Path, py: Optional[str], base_url: str = HF) -> None:
    spec = model_spec(m, model)
    d = root / model
    d.mkdir(parents=True, exist_ok=True)
    print(f"fetch {model}: {spec['repo']} @ {spec['revision'][:12]} -> {d}", flush=True)
    for name, want in spec["files"].items():
        p = d / name
        if p.exists() and sha256_file(p) == want:
            print(f"  {name}: present, SHA-256 matches")
            continue
        t0 = time.time()
        download(f"{base_url}/{spec['repo']}/resolve/{spec['revision']}/{name}", p, want)
        print(f"  {name}: downloaded, SHA-256 matches ({time.time() - t0:.0f} s)", flush=True)
    missing = []
    for name, t in m["texts"].items():
        p = d / name
        if p.exists() and sha256_file(p) == t["sha256"]:
            print(f"  {name}: present, SHA-256 matches")
            continue
        src = next((q for q in sorted(root.glob(f"*/{name}"))
                    if q.parent != d and sha256_file(q) == t["sha256"]), None)
        if src is not None:
            shutil.copy2(src, p)
            print(f"  {name}: copied from {src.parent.name} (SHA-256 matches)")
        else:
            missing.append(name)
    if missing:                                     # llm_study.py fetch writes both texts
        run_study(study_python(py), ["fetch", "--assets", str(d)])
        for name in m["texts"]:
            got = sha256_file(d / name)
            if got != m["texts"][name]["sha256"]:
                raise CalibrationError(f"{name}: fetched text has SHA-256 {got}, pinned "
                                       f"{m['texts'][name]['sha256']} — the dataset changed")
            print(f"  {name}: fetched, SHA-256 matches")


def cmd_calibrate(m: dict, model: str, root: Path, py: Optional[str], *, check: bool = False,
                  record: bool = False, force: bool = False, manifest_path: Path = MANIFEST) -> str:
    """Calibrate into a scratch file and compare it with the manifest (and the
    installed file).  ``calibrate`` installs the result — with its provenance —
    when it reproduces the manifest, with ``record`` (the manifest takes its
    hash) or ``force``; otherwise the installed file stays and the result is
    kept next to it as formats_<policy>.new.json.  ``check`` never installs.
    Returns "reproduced", "recorded", "unrecorded", "differs" or "stale"
    (reproduced, but the installed file is another one)."""
    spec = model_spec(m, model)
    policy = spec["formats"]["policy"]
    inputs = verify_inputs(m, model, root)
    py = study_python(py)
    installed = study_dir(root, model) / f"formats_{policy}.json"
    tmp = Path(tempfile.mkdtemp(prefix="llm_calibrate_"))
    try:
        out = tmp / installed.name
        print(f"{'check' if check else 'calibrate'} {model}: policy {policy}", flush=True)
        t0 = time.time()
        run_study(py, ["formats", "--assets", str(root / model), "--base", policy, "--out", str(out)])
        got = sha256_file(out)
        want = spec["formats"].get("sha256")
        if want is None:
            status = "unrecorded"
        else:
            status = "reproduced" if got == want else "differs"
        print(f"  formats SHA-256 {got[:16]}…  manifest {str(want)[:16]}…  -> {status.upper()} "
              f"({time.time() - t0:.0f} s)", flush=True)
        same_installed = installed.exists() and sha256_file(installed) == got
        if installed.exists() and not same_installed:
            print(f"  differs from the installed {installed}:")
            for line in diff_formats(installed, out):
                print("    " + line)
        if check:
            if status == "reproduced" and installed.exists() and not same_installed:
                status = "stale"
            return status
        if record and status != "reproduced":
            spec["formats"]["sha256"] = got
            save_manifest(m, manifest_path)
            status = "recorded"
        if status == "differs" and not force:
            new = installed.with_name(installed.stem + ".new.json")
            new.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(out), new)
            print(f"  not installed (the result is {new}); --force installs it, "
                  f"--record also stores its hash in {manifest_path.name}")
            return status
        installed.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(out), installed)
        prov = {"model": model, "repo": spec["repo"], "revision": spec["revision"],
                "policy": policy, "formats": installed.name, "formats_sha256": got,
                "manifest_sha256": spec["formats"].get("sha256"), "status": status,
                "inputs": inputs, "code": code_version(), "environment": environment(py),
                "command": ["llm_study.py", "formats", "--assets", f"assets/{model}",
                            "--base", policy],
                "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        pp = installed.with_name(installed.stem + ".provenance.json")
        with open(pp, "w") as f:
            json.dump(prov, f, indent=2)
            f.write("\n")
        print(f"  installed {installed}\n  provenance {pp}")
        if status == "unrecorded":
            print(f"  the manifest has no hash for it yet: `calibrate {model} --record`")
        return status
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def study_metrics(results_json: Path, policy: str = SHIPPED) -> dict:
    """The shipped policy's headline numbers from llm_study.py study's results."""
    r = json.loads(Path(results_json).read_text())["results"]
    a = r[policy]["agree"]
    return {"policy": policy, "ppl": round(r[policy]["ppl"], 6), "float_ppl": round(r["float"]["ppl"], 6),
            "top1_all": round(a["tf_all"] / a["tf_all_n"], 6),
            "top1_resp": round(a["tf_resp"] / a["tf_resp_n"], 6),
            "top1_held": round(a["h_all"] / a["h_all_n"], 6),
            "kl_held": round(a["h_kl"] / a["h_all_n"], 6), "identical": r[policy]["identical"]}


def cmd_study(m: dict, model: str, root: Path, py: Optional[str], *, quick: bool = False,
              record: bool = False, manifest_path: Path = MANIFEST) -> str:
    spec = model_spec(m, model)
    verify_inputs(m, model, root)
    out = study_dir(root, model) / ("shipped_quick" if quick else "shipped")
    args = ["study", "--assets", str(root / model), "--policies", SHIPPED_POLICIES, "--out", str(out)]
    run_study(study_python(py), args + (["--quick"] if quick else []))
    got = study_metrics(out / "results.json")
    print(f"study {model}: {json.dumps(got)}")
    want = spec.get("study")
    if record and not quick:
        spec["study"] = got
        save_manifest(m, manifest_path)
        return "recorded"
    if quick or want is None:
        return "unrecorded"
    status = "reproduced" if got == want else "differs"
    if status == "differs":
        print("  manifest: " + json.dumps(want))
    print(f"  -> {status.upper()}")
    return status


def cmd_add(m: dict, model: str, repo: str, revision: str, root: Path,
            manifest_path: Path = MANIFEST, base_url: str = HF) -> None:
    with urllib.request.urlopen(f"{base_url}/api/models/{repo}/revision/{revision}", timeout=60) as r:
        info = json.load(r)
    sha = info["sha"]
    d = root / model
    files = {}
    for name in CHECKPOINT_FILES:
        p = d / name
        files[name] = download(f"{base_url}/{repo}/resolve/{sha}/{name}", p, None)
        print(f"  {name}: {files[name][:16]}…", flush=True)
    m["models"][model] = {"repo": repo, "revision": sha,
                          "license": (info.get("cardData") or {}).get("license"),
                          "files": files, "formats": {"policy": "pow2+sink+p12", "sha256": None}}
    save_manifest(m, manifest_path)
    print(f"add {model}: {repo} @ {sha} pinned in {manifest_path}; next: "
          f"`fetch {model}` (texts), `calibrate {model} --record`")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n", 2)[2])
    ap.add_argument("command", choices=("fetch", "calibrate", "check", "study", "validate", "all", "add"))
    ap.add_argument("model")
    ap.add_argument("--assets-root", default=str(ASSETS), help="default demo/chat/assets")
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--study-python", default=None)
    ap.add_argument("--record", action="store_true",
                    help="calibrate / study: store the result in the manifest")
    ap.add_argument("--force", action="store_true",
                    help="calibrate: install the result even when it differs from the manifest")
    ap.add_argument("--quick", action="store_true", help="study: 4 prompts, 1 short window")
    ap.add_argument("--study", action="store_true", help="all: also run the study")
    ap.add_argument("--validate", action="store_true", help="all: also validate against torch")
    ap.add_argument("--repo", default=None, help="add: the Hugging Face repository")
    ap.add_argument("--revision", default="main", help="add: branch, tag or commit to pin")
    args = ap.parse_args(argv)
    mpath, root = Path(args.manifest), Path(args.assets_root)
    m = load_manifest(mpath)
    try:
        if args.command == "add":
            if not args.repo:
                ap.error("add needs --repo")
            cmd_add(m, args.model, args.repo, args.revision, root, mpath)
            return 0
        if args.command in ("fetch", "all"):
            cmd_fetch(m, args.model, root, args.study_python)
        if args.command == "validate" or (args.command == "all" and args.validate):
            verify_inputs(m, args.model, root)
            run_study(study_python(args.study_python), ["validate", "--assets", str(root / args.model)])
        status = None
        if args.command in ("calibrate", "check", "all"):
            status = cmd_calibrate(m, args.model, root, args.study_python,
                                   check=args.command == "check", record=args.record,
                                   force=args.force, manifest_path=mpath)
        if args.command == "study" or (args.command == "all" and args.study):
            s = cmd_study(m, args.model, root, args.study_python, quick=args.quick,
                          record=args.record, manifest_path=mpath)
            status = status if status in ("differs", "stale") else s
        return 1 if status in ("differs", "stale") else 0
    except CalibrationError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
