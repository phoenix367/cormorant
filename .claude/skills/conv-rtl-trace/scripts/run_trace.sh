#!/usr/bin/env bash
# run_trace.sh <build-dir> <fixture-dir> <log-path>
# Run the conv testbench on one fixture with the +VERBOSE probes enabled.
set -euo pipefail
build=$(cd "${1:?build dir}" && pwd); fx=$(cd "${2:?fixture dir}" && pwd); log=${3:?log path}
root=$(cd "$(dirname "$0")/../../../.." && pwd)
ip="$build/rtl_ip/ConvKernel_ip"
[ -d "$ip" ] || { echo "IP repo $ip missing — run: make -C $build package_conv_rtl" >&2; exit 1; }
report="${log%.log}_report.json"
t0=$(date +%s)
TS_VERBOSE=1 make -C "$root/hw/cormorant_test_stand" tb-conv \
    DATA_DIR_conv="$fx" REPORT_conv="$report" IP_REPO_conv="$ip" > "$log" 2>&1 || true
echo "wall $(( $(date +%s) - t0 )) s; log $log; report $report"
grep -E "^\[ts\] kernel=|\[SCB\] (PASS|FAIL)|forward_progress|Pending AW" "$log" | cut -c1-120 | tail -5
