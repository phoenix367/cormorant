#!/usr/bin/env python3
"""rpu_board.py — run a generated project with its kernels driven by the KV260's
Cortex-R5 (doc/plans/RPU_PLAN.md track B1) and check it against the same
library driving them from the A53.

  python rpu_board.py --config remote_config.json --bsp-dir <bsp> --project <dir> [--project ...]
  python rpu_board.py --config remote_config.json --bsp-dir <bsp> --start-only   (firmware up, left running)
  python rpu_board.py --config remote_config.json --stop                         (firmware stopped)

Steps (the per-board lock is held throughout, src/remote/lock.py):
  firmware -> cmake + make of runtime/rpu (the model-independent firmware: it
              replays the command stream the library records) with the R5
              toolchain and the BSP tools/rpu/make_r5_bsp.py made from the
              bitstream's XSA
  upload   -> the firmware (/lib/firmware/<name>.elf), the overlay
              (board/kv260/rpu/rpu.dts compiled with dtc) and the projects
  rpu      -> the overlays applied once (configfs: rpu.dts, and rpu-ipi.dts — the
              A53's doorbell, the rpu_ipi UIO device), zynqmp_r5_remoteproc
              loaded, remoteproc0 started on the firmware (restarted if it was
              running)
  per project:
    build  -> its Linux library and rpu/rpu_host.c on the board, with the
              config's UIO names
    run    -> rpu_host: N inferences with ramp inputs, each run first with the
              A53 driving the kernels (the reference), then with the R5 driving
              them (--mode r5: the firmware replays the recorded kernel calls with
              interrupt-driven waits) or the project's policy choosing (--mode
              auto: small models stay on the A53, src/rpu_policy.py); compared
              bit for bit, both timed, the R5's counters; --mode both (default)
              runs one pass of each
  stop     -> remoteproc0 stopped (the overlay stays); the board's CMA cleaned
              (src/remote/cma.py: page cache dropped, memory compacted)

Exit 0: every project bit-exact; 1: a mismatch or a firmware error; 2: a
setup error.  The chat server must be stopped first (demo/chat/deploy.py
--stop): it owns the kernels the firmware drives.
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.remote import RemoteSession, hold_board_lock, load_config, uio_devices_from_cfg   # noqa: E402
from src.remote.cma import clean_cma, describe                                     # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
FIRMWARE_SRC = HERE / "runtime" / "rpu"
DTS = REPO / "board" / "kv260" / "rpu" / "rpu.dts"
DTS_IPI = REPO / "board" / "kv260" / "rpu" / "rpu-ipi.dts"       # the A53's doorbell (rpu_ipi UIO)
VITIS = Path(os.environ.get("XILINX_VITIS", "/mnt/data/xilinx/2025.2/Vitis"))


def build_firmware(bsp_dir: Path, build_dir: Path, log) -> Path:
    cmd = ["cmake", "-S", str(FIRMWARE_SRC), "-B", str(build_dir),
           f"-DCMAKE_TOOLCHAIN_FILE={FIRMWARE_SRC / 'toolchain-r5.cmake'}", f"-DBSP_DIR={bsp_dir}",
           f"-DVITIS_DIR={VITIS}"]
    subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
    subprocess.run(["cmake", "--build", str(build_dir), "-j4"], check=True, stdout=log, stderr=subprocess.STDOUT)
    elf = build_dir / "rpu_firmware.elf"
    if not elf.is_file():
        raise RuntimeError("the firmware did not build (see the log)")
    return elf


def compile_overlay(out: Path, dts: Path = DTS) -> Path:
    dtc = VITIS / "bin" / "dtc"
    if not dtc.is_file():
        dtc = Path("dtc")
    dtbo = out / (dts.stem + ".dtbo")
    subprocess.run([str(dtc), "-I", "dts", "-O", "dtb", "-@", "-o", str(dtbo), str(dts)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return dtbo


FIRMWARE_STOP = ("R=$(ls -d /sys/class/remoteproc/remoteproc* 2>/dev/null | head -1); "
                 "[ -n \"$R\" ] && echo stop > $R/state 2>/dev/null; [ -n \"$R\" ] && cat $R/state; true")


def build_artifacts(bsp_dir: Path, out: Path, log) -> tuple:
    """(firmware ELF, rpu.dtbo, rpu-ipi.dtbo) built under ``out`` (the firmware
    build's output goes to ``log``)."""
    return (build_firmware(Path(bsp_dir).resolve(), out / "build-rpu", log),
            compile_overlay(out), compile_overlay(out, DTS_IPI))


def firmware_up_script(work: str, firmware_name: str = "rpu_firmware.elf") -> str:
    """The board-side shell: the firmware into /lib/firmware, the overlays applied
    once (configfs: rpu.dtbo, rpu-ipi.dtbo from ``work``), zynqmp_r5_remoteproc
    loaded, remoteproc0 (re)started on the firmware; prints its state."""
    return f"""
set -e
cp {work}/{firmware_name} /lib/firmware/{firmware_name}
modprobe zynqmp_r5_remoteproc 2>/dev/null || true
if [ ! -d /sys/kernel/config/device-tree/overlays/rpu ]; then
    mkdir /sys/kernel/config/device-tree/overlays/rpu
    cat {work}/rpu.dtbo > /sys/kernel/config/device-tree/overlays/rpu/dtbo
fi
cat /sys/kernel/config/device-tree/overlays/rpu/status
if [ ! -d /sys/kernel/config/device-tree/overlays/rpu-ipi ]; then
    mkdir /sys/kernel/config/device-tree/overlays/rpu-ipi
    cat {work}/rpu-ipi.dtbo > /sys/kernel/config/device-tree/overlays/rpu-ipi/dtbo
fi
echo "doorbell: $(cat /sys/kernel/config/device-tree/overlays/rpu-ipi/status) \
$(grep -l rpu_ipi /sys/class/uio/uio*/name 2>/dev/null | head -1)"
R=$(ls -d /sys/class/remoteproc/remoteproc* | head -1); echo "remoteproc $R"
echo stop > $R/state 2>/dev/null || true
sleep 1
echo {firmware_name} > $R/firmware
echo start > $R/state
sleep 1
cat $R/state
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="remote_config.json")
    ap.add_argument("--project", action="append", default=[],
                    help="a generated project (DMA inputs and outputs); repeatable")
    ap.add_argument("--start-only", action="store_true",
                    help="build, load and start the firmware (and the overlays), gate nothing, leave "
                         "it running: the libraries on the board then attach by themselves")
    ap.add_argument("--stop", action="store_true", help="stop the firmware (remoteproc) and exit")
    ap.add_argument("--bsp-dir", help="the R5 standalone BSP (tools/rpu/make_r5_bsp.py)")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--ring-kb", type=int, default=0, help="the command ring (default 1 MiB)")
    ap.add_argument("--firmware-name", default="rpu_firmware.elf")
    ap.add_argument("--no-check", action="store_true", help="time the R5 only, no A53 reference")
    ap.add_argument("--mode", choices=("r5", "auto", "both"), default="both",
                    help="r5: the R5 drives every checked run (the gate of the R5 path); auto: the "
                         "project's policy decides (INFERENCE_RPU_RECOMMENDED, src/rpu_policy.py); "
                         "both (default): one pass of each")
    ap.add_argument("--keep-running", action="store_true", help="leave the firmware running")
    ap.add_argument("--board-lock", default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.stop:
        hold_board_lock(cfg, args.board_lock)
        session = RemoteSession(cfg["ssh"])
        session.connect()
        try:
            out, _err, _rc = session.exec(FIRMWARE_STOP, timeout=30)
            print(f"rpu: {out.strip() or 'no remoteproc'}")
        finally:
            session.close()
        return 0
    if not args.project and not args.start_only:
        ap.error("--project is required (or --start-only / --stop)")
    if not args.bsp_dir:
        ap.error("--bsp-dir is required to build the firmware")
    if args.start_only:
        args.keep_running = True
    projects = [Path(p).resolve() for p in args.project]
    for project in projects:
        if not (project / "rpu" / "rpu_host.c").is_file() or not (project / "src" / "inference_rpu.c").is_file():
            print(f"{project}: no rpu/rpu_host.c or src/inference_rpu.c (generate the project again)", file=sys.stderr)
            return 2
    hold_board_lock(cfg, args.board_lock)

    with tempfile.TemporaryDirectory(prefix="rpu_board_") as td:
        tdp = Path(td)
        print("firmware: building for the R5 ...", flush=True)
        with open(tdp / "firmware_build.log", "w") as log:
            try:
                elf, dtbo, dtbo_ipi = build_artifacts(Path(args.bsp_dir), tdp, log)
            except (subprocess.CalledProcessError, RuntimeError) as e:
                print(f"firmware: build failed: {e}\n{(tdp / 'firmware_build.log').read_text()[-3000:]}",
                      file=sys.stderr)
                return 2
        print(f"firmware: {elf.stat().st_size} bytes; overlay {dtbo.name}", flush=True)

        session = RemoteSession(cfg["ssh"])
        session.connect()
        try:
            work = cfg["remote"]["work_dir"].rstrip("/") + "/rpu_board"
            session.exec(f"rm -rf {work} && mkdir -p {work}")
            print(f"upload: {len(projects)} project(s), the firmware and the overlay ...", flush=True)
            stage = tdp / "stage"
            stage.mkdir()
            names = []
            for project in projects:
                name = project.name
                while name in names:
                    name += "_"
                names.append(name)
                subprocess.run(["cp", "-a", str(project), str(stage / name)], check=True)
                subprocess.run(["rm", "-rf", str(stage / name / "build")])
            subprocess.run(["cp", str(elf), str(stage / args.firmware_name)], check=True)
            subprocess.run(["cp", str(dtbo), str(stage / "rpu.dtbo")], check=True)
            subprocess.run(["cp", str(dtbo_ipi), str(stage / "rpu-ipi.dtbo")], check=True)
            session.upload_dir(stage, work)

            script = firmware_up_script(work, args.firmware_name)
            out, err, rc = session.exec(script, timeout=60)
            if args.verbose or rc != 0:
                print(out + err)
            if rc != 0 or "running" not in out:
                print("rpu: the firmware did not start (dmesg on the board tells why)", file=sys.stderr)
                return 2
            print("rpu: firmware running", flush=True)
            # the uploads' page cache sits in the CMA area: clean it before any library
            # allocates its pool (a 311 MB pool failed with 880 MB free once, 2026-10-13)
            print(f"cma: {describe(clean_cma(session)['after'])}", flush=True)
            if args.start_only:
                return 0

            flags = " ".join(f'-DINFERENCE_{k.upper()}_INSTANCE=\\"{v}\\"' for k, v in uio_devices_from_cfg(cfg).items())
            flags += " " + " ".join(cfg["remote"].get("cmake_args", []))
            results = {}
            for name in names:
                pdir = f"{work}/{name}"
                print(f"\n== {name}\nbuild: the Linux library and rpu_host on the board ...", flush=True)
                out, err, rc = session.exec(
                    f"cd {pdir} && cmake -S . -B build -DINFERENCE_TARGET=LINUX {flags} > cmake.log 2>&1 "
                    f"&& make -C build -j1 inference rpu_host > make.log 2>&1", timeout=cfg["build"]["timeout"] * 4)
                if rc != 0:
                    out2, _, _ = session.exec(f"tail -30 {pdir}/cmake.log {pdir}/make.log")
                    print(f"build failed:\n{out2}", file=sys.stderr)
                    results[name] = 2
                    continue
                worst_rc = 0
                for mode in (("r5", "auto") if args.mode == "both" else (args.mode,)):
                    print(f"run: {args.runs} inferences, "
                          + ("the R5 driving the kernels" if mode == "r5" else "the policy choosing the driver")
                          + " ...", flush=True)
                    cmd = f"cd {pdir} && timeout {cfg['run']['timeout']} ./build/rpu_host --runs {args.runs}" \
                          + (f" --ring-kb {args.ring_kb}" if args.ring_kb else "") \
                          + (" --no-check" if args.no_check else "") + (" --auto" if mode == "auto" else "")
                    out, err, rc = session.exec(cmd, timeout=cfg["run"]["timeout"] + 30)
                    lines = out.strip().splitlines()
                    shown = lines if args.verbose else [ln for ln in lines if not ln.startswith("run ") or "MISMATCH" in ln][:12]
                    print("\n".join(shown))
                    if err.strip():
                        print(err.strip(), file=sys.stderr)
                    worst_rc = max(worst_rc, rc)
                results[name] = worst_rc
            if not args.keep_running:
                session.exec(FIRMWARE_STOP, timeout=30)
            # the projects' pools fragment the CMA: clean it for whatever comes next
            # (the chat server's 775 MiB pool failed to allocate after six gates once)
            clean_cma(session)
            print()
            for name, rc in results.items():
                print(f"BOARD GATE {name}: " + ("bit-exact with the A53" if rc == 0 else
                                                 "MISMATCH" if rc == 1 else f"failed (rc {rc})"))
            worst = max(results.values()) if results else 2
            return 0 if worst == 0 else (1 if worst == 1 else 2)
        finally:
            session.close()


if __name__ == "__main__":
    sys.exit(main())
