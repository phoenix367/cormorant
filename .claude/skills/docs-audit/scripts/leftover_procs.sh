#!/usr/bin/env bash
# leftover_procs.sh — find (and with --kill, stop) host processes a
# reproduction agent left behind (docs-audit mode B, step 5): every process of
# this user whose command line mentions DIR or whose cwd is under DIR, plus
# the children of those (a polling loop's sleep, a make's xsim).  Shell
# polling loops ('until …; do sleep' / 'while …; do sleep') that do NOT
# mention DIR are listed separately and never killed — another session may
# own them.  On 2026-09-29 six loops outlived the agent by hours, waiting for
# an 'rc=' line a log never got.
#
# usage: leftover_procs.sh [--kill] DIR [DIR...]
#   rc 0: nothing under DIR; rc 1: leftovers listed (without --kill)
#   --kill sends TERM, then KILL to survivors after 5 s.  This script and its
#   ancestors (the calling shell) are never matched.
set -euo pipefail

KILL=0
[ "${1:-}" = --kill ] && { KILL=1; shift; }
[ $# -ge 1 ] || { echo "usage: leftover_procs.sh [--kill] DIR [DIR...]" >&2; exit 2; }
DIRS=()
for d in "$@"; do DIRS+=("$(realpath -m "$d")"); done

ppid_of() { awk '{print $4}' "/proc/$1/stat" 2>/dev/null || true; }

declare -A SKIP=()            # this script and its ancestors
p=$$
while [ -n "$p" ] && [ "$p" -gt 1 ]; do SKIP[$p]=1; p=$(ppid_of "$p"); done

declare -A HIT=() LOOP=()
for proc in /proc/[0-9]*; do
    pid=${proc#/proc/}
    [ -n "${SKIP[$pid]:-}" ] && continue
    [ -O "$proc" ] || continue                        # this user's processes only
    [ "$(ppid_of "$pid")" = "$$" ] && continue        # this script's own ps / awk
    cmd=$(tr '\0' ' ' < "$proc/cmdline" 2>/dev/null || true)
    [ -n "$cmd" ] || continue
    cwd=$(readlink "$proc/cwd" 2>/dev/null || true)
    hit=
    for d in "${DIRS[@]}"; do
        case "$cmd" in *"$d"*) hit=1 ;; esac
        case "$cwd/" in "$d"/*) hit=1 ;; esac
    done
    if [ -n "$hit" ]; then
        HIT[$pid]=1
    elif [[ "$cmd" =~ (^|/)(ba|z)?sh\ .*(until|while)\ .*sleep ]]; then
        LOOP[$pid]=1
    fi
done
# descendants of a match are leftovers too (repeat to cover grandchildren)
for _ in 1 2 3 4; do
    for proc in /proc/[0-9]*; do
        pid=${proc#/proc/}
        [ -n "${HIT[$pid]:-}" ] || [ -n "${SKIP[$pid]:-}" ] && continue
        pp=$(ppid_of "$pid")
        [ -n "$pp" ] && [ -n "${HIT[$pp]:-}" ] && [ -O "$proc" ] && HIT[$pid]=1
    done
done

show() {   # show PID... as a forest
    ps -eo pid,ppid,etime,args --forest |
        awk -v ids="$*" 'BEGIN { n = split(ids, a, " "); for (i = 1; i <= n; i++) s[a[i]] = 1 }
                         NR == 1 || ($1 in s)' | sed 's/^/   /'
}

if [ ${#LOOP[@]} -gt 0 ]; then
    echo "polling loops NOT under ${DIRS[*]} (not killed; check by hand):"
    show "${!LOOP[@]}"
fi
if [ ${#HIT[@]} -eq 0 ]; then
    echo "no leftover processes under: ${DIRS[*]}"
    exit 0
fi
echo "leftover processes under ${DIRS[*]} (${#HIT[@]}):"
show "${!HIT[@]}"

[ "$KILL" = 1 ] || { echo "(re-run with --kill to stop them)"; exit 1; }
kill -TERM "${!HIT[@]}" 2>/dev/null || true
sleep 5
alive=()
for pid in "${!HIT[@]}"; do kill -0 "$pid" 2>/dev/null && alive+=("$pid"); done
if [ ${#alive[@]} -gt 0 ]; then
    echo "KILL ${alive[*]}"
    kill -KILL "${alive[@]}" 2>/dev/null || true
fi
echo "stopped ${#HIT[@]} process(es)"
