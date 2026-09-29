#!/usr/bin/env bash
# board_snapshot.sh — READ-ONLY snapshot of the KV260's production state,
# taken before a fresh-clone reproduction and again after the restore
# (docs-audit mode B, steps 3 and 5).
#
# usage:
#   board_snapshot.sh OUT_FILE                 snapshot the board over ssh
#   board_snapshot.sh --local OUT_FILE         run the same commands on this host (script test)
#   board_snapshot.sh --compare BEFORE AFTER   diff two snapshots; rc 1 when a stable line differs
#
# env: BOARD_HOST (192.168.100.8)  BOARD_USER (root)  BOARD_KEY (~/.ssh/kv260-testkey)
#
# Lines starting with '~' are volatile (uptime, free space, CMA) and are shown
# side by side by --compare, not diffed.  Stable lines: firmware SHA-256
# (pl.bin = the production bitstream id), overlays + status, UIO names,
# /root listing, repro leftovers, chat service, tmpfiles rule, cpuidle, cma.
# Nothing here writes on the board.
set -euo pipefail

HOST=${BOARD_HOST:-192.168.100.8}
USER_=${BOARD_USER:-root}
KEY=${BOARD_KEY:-$HOME/.ssh/kv260-testkey}

die() { echo "board_snapshot: $*" >&2; exit 2; }

if [ "${1:-}" = --compare ]; then
    [ $# -eq 3 ] || die "usage: --compare BEFORE AFTER"
    A=$2; B=$3
    [ -f "$A" ] && [ -f "$B" ] || die "missing snapshot file"
    echo "== volatile (before | after)"
    paste -d'|' <(grep '^~' "$A") <(grep '^~' "$B") | sed 's/^/   /'
    echo "== stable lines that changed"
    if diff <(grep -v '^[~#]' "$A") <(grep -v '^[~#]' "$B"); then
        echo "   none — board state restored"
        exit 0
    fi
    exit 1
fi

LOCAL=0
if [ "${1:-}" = --local ]; then LOCAL=1; shift; fi
OUT=${1:-}
[ -n "$OUT" ] || die "usage: board_snapshot.sh [--local] OUT_FILE | --compare BEFORE AFTER"

# The remote side: read-only commands only (ls, cat, sha256sum, md5sum, df,
# systemctl is-*, grep, ps).  Fed to 'bash -s' on stdin.
REMOTE=$(cat <<'EOS'
LC_ALL=C
echo "# board snapshot $(date -Is) $(hostname)"
for f in /lib/firmware/*.bin /lib/firmware/*.dtbo; do
    [ -e "$f" ] && echo "firmware: $(sha256sum "$f" | cut -c1-64) $f"
done
echo "firmware_apps: $(ls /lib/firmware/xilinx 2>/dev/null | tr '\n' ' ')"
ov=/sys/kernel/config/device-tree/overlays
if [ -d "$ov" ]; then
    for d in "$ov"/*/; do
        [ -d "$d" ] || continue
        echo "overlay: $(basename "$d") status=$(cat "$d/status" 2>/dev/null) path=$(cat "$d/path" 2>/dev/null)"
    done
else
    echo "overlay: (no configfs overlays dir)"
fi
echo "uio: $(for f in /sys/class/uio/uio*/name; do [ -e "$f" ] && printf '%s=%s ' "$(basename "$(dirname "$f")")" "$(cat "$f")"; done)"
echo "root_entries: $(ls -A /root 2>/dev/null | tr '\n' ' ')"
echo "repro_leftovers: $(ls -d /root/*repro* /tmp/*repro* /lib/firmware/design_cormorant* /tmp/design_cormorant* $ov/design_cormorant 2>/dev/null | tr '\n' ' ')"
if command -v systemctl >/dev/null; then
    echo "chat_service: active=$(systemctl is-active kv260-chat 2>/dev/null) enabled=$(systemctl is-enabled kv260-chat 2>/dev/null)"
fi
echo "server_procs: $(ps -eo args 2>/dev/null | grep -E '[k]v260_chat_server|[t]est_inference|[c]lassify_images|[b]ert_squad( |$)|[k]ernel_perf' | sort | tr '\n' ';')"
rule=/etc/tmpfiles.d/kv260-no-cpu-powerdown.conf
echo "tmpfiles_rule: $( [ -e "$rule" ] && md5sum "$rule" | cut -c1-32 || echo missing)"
echo "cpuidle_state1_disable: $(cat /sys/devices/system/cpu/cpu*/cpuidle/state1/disable 2>/dev/null | tr '\n' ' ')"
echo "cmdline_cma: $(grep -o 'cma=[^ ]*' /proc/cmdline || echo none)"
echo "~uptime: $(uptime | sed 's/^ *//')"
echo "~df_root: $(df -h / | tail -1 | awk '{print $4" free of "$2}')"
echo "~df_tmp: $(df -h /tmp | tail -1 | awk '{print $4" free of "$2" ("$6")"}')"
echo "~mem: $(grep -E '^(MemAvailable|CmaTotal|CmaFree):' /proc/meminfo | tr -s ' ' | tr '\n' ' ')"
EOS
)

mkdir -p "$(dirname "$OUT")"
if [ "$LOCAL" = 1 ]; then
    bash -s <<<"$REMOTE" > "$OUT"
else
    [ -r "$KEY" ] || die "ssh key not readable: $KEY"
    # BatchMode: never prompt; timeout: an unresponsive board fails fast
    timeout 60 ssh -i "$KEY" -o BatchMode=yes -o ConnectTimeout=10 \
        "$USER_@$HOST" 'bash -s' <<<"$REMOTE" > "$OUT" ||
        die "ssh to $USER_@$HOST failed or timed out (rc $?) — board unresponsive?"
fi
cat "$OUT"
