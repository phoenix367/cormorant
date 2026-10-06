#!/bin/bash
# ---------------------------------------------------------------------------
# run.sh <kernel> [rtl_dir] [work_dir] — netlist equivalence of one RTL kernel
# (matmul | vectorop | pool): Vivado synthesises the kernel top out of context
# (the options of syn/synth_ooc.tcl), writes the functional netlist, and xsim
# runs the RTL and the netlist in lockstep (gen_tb.py, jobs_<kernel>.svh).
# Exit status 1 on any mismatch.  It catches what Verilator cannot: RTL that
# Vivado synthesises differently from how it simulates.
# ---------------------------------------------------------------------------
set -e
K=$1
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
R=$(realpath "${2:-$REPO/kernels/${K}_rtl/rtl}")
D=$(realpath -m "${3:-$PWD/neteq_$K}")
case $K in
  matmul)   TOP=MatmulKernel;;
  vectorop) TOP=VectorOPKernel;;
  pool)     TOP=PoolingKernel;;
  *) echo "usage: run.sh matmul|vectorop|pool [rtl_dir] [work_dir]"; exit 2;;
esac
mkdir -p "$D"; cd "$D"; rm -f net_raw.v
ORDER=$(grep -o "rtl/[a-zA-Z_0-9]*\.sv" "$REPO/kernels/${K}_rtl/syn/rtl_files.tcl" | sed 's|rtl/||')
SV=""; for f in $ORDER; do [ -f "$R/$f" ] && SV="$SV $R/$f"; done
cat > synth.tcl <<TCL
set_param general.maxThreads 4
read_verilog -sv [list $SV]
read_verilog $R/$TOP.v
synth_design -top $TOP -part xck26-sfvc784-2LV-c -mode out_of_context -flatten_hierarchy rebuilt -directive PerformanceOptimized
write_verilog -mode funcsim -force net_raw.v
TCL
vivado -mode batch -nojournal -log synth.log -source synth.tcl > /dev/null
python3 - <<'PY'
import re
s = open('net_raw.v').read()
for m in sorted(set(re.findall(r'^module (\w+)', s, re.M)), key=len, reverse=True):
    if m != 'glbl':
        s = re.sub(r'\b%s\b' % re.escape(m), 'N_' + m, s)
open('net.v', 'w').write(s)
PY
python3 "$HERE/gen_tb.py" "$R/$TOP.v" "$HERE/jobs_$K.svh" tbk.sv
xvlog -sv --relax $SV tbk.sv > xvlog.log 2>&1
xvlog --relax "$R/$TOP.v" net.v >> xvlog.log 2>&1
xelab --relax -L unisims_ver -L secureip tbk glbl -s sim -timescale 1ns/1ps > xelab.log 2>&1
xsim sim -R > xsim.log 2>&1 || true
grep -E "^job|DONE|MISMATCH" xsim.log | head -60
grep -q "DONE errors=0 " xsim.log
