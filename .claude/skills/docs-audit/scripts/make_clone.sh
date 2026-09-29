#!/usr/bin/env bash
# make_clone.sh — docs-audit mode B, steps 1–2: a fresh clone exactly as the
# README Quick start §1 says, with the maintainer's unpushed work overlaid as
# local commits (so the reproduction tests the CURRENT docs).
#
# usage:
#   make_clone.sh --list-untracked MAIN_REPO
#       print MAIN_REPO's untracked, not-ignored paths with sizes, then exit;
#       put the ones that belong to the work into a list file for --files
#   make_clone.sh [--files LIST] [--root DIR] [--url URL] [--min-free-gb N] MAIN_REPO DATE
#
#   --files LIST     untracked paths to copy (one per line, relative to
#                    MAIN_REPO, '#' comments allowed; a directory means its
#                    untracked, not-ignored files).  Without it no untracked
#                    file is copied.
#   --root DIR       parent of the reproduction dir (default /mnt/data)
#   --url URL        clone URL (default git@github.com:phoenix367/cormorant.git)
#   --min-free-gb N  free space required on DIR's filesystem (default 10)
#
# Creates DIR/cormorant_repro_DATE/ with
#   cormorant/          the clone (+ submodules), origin's HEAD plus up to two
#                       local commits: "local: unpushed working-copy changes
#                       (not for push)" and "local: unpushed new files";
#                       push URLs disabled
#   overlay.patch       git diff <clone HEAD> of MAIN_REPO, submodules excluded
#   overlay_files.txt   the untracked files copied
#   clone_info.txt      commits, sizes, what was overlaid
#   env.sh              TMPDIR / XDG_CACHE_HOME / PIP_CACHE_DIR / HF_HOME /
#                       PIP_CONFIG_FILE for every shell of the reproduction
#   brief.md, GAPS.md   templates/new_user_brief.md and GAPS_template.md with
#                       the placeholders filled (env BOARD_HOST, BOARD_KEY,
#                       XILINX_ROOT override the board / Xilinx defaults)
#   tmp/ cache/ logs/
# Never writes to MAIN_REPO.  Never pushes.
set -euo pipefail

URL=git@github.com:phoenix367/cormorant.git
ROOT=/mnt/data
MIN_FREE_GB=10
FILES=
LIST_ONLY=0

die() { echo "make_clone: $*" >&2; exit 1; }
usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --list-untracked) LIST_ONLY=1; shift ;;
        --files)          FILES=${2:?}; shift 2 ;;
        --root)           ROOT=${2:?}; shift 2 ;;
        --url)            URL=${2:?}; shift 2 ;;
        --min-free-gb)    MIN_FREE_GB=${2:?}; shift 2 ;;
        -h|--help)        usage 0 ;;
        -*)               die "unknown option $1" ;;
        *)                break ;;
    esac
done

MAIN=${1:-}
[ -n "$MAIN" ] || usage 2
MAIN=$(cd "$MAIN" && pwd) || die "no such directory: $1"
git -C "$MAIN" rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "$MAIN is not a git work tree"
[ "$(git -C "$MAIN" rev-parse --show-toplevel)" = "$MAIN" ] || die "$MAIN is not the top of its repository"

if [ "$LIST_ONLY" = 1 ]; then
    echo "# untracked, not-ignored paths of $MAIN (size  path); copy the ones that belong"
    echo "# to the work into a list file.  Not: models, assets, build trees, venvs, locks."
    git -C "$MAIN" status --porcelain=v1 -z --untracked-files=normal --ignore-submodules=all |
        while IFS= read -r -d '' e; do
            [ "${e:0:3}" = "?? " ] || continue
            p=${e:3}
            printf '%8s  %s\n' "$(du -sh "$MAIN/$p" 2>/dev/null | cut -f1)" "$p"
        done
    exit 0
fi

DATE=${2:-}
[ -n "$DATE" ] || usage 2
[[ "$DATE" =~ ^[A-Za-z0-9._-]+$ ]] || die "DATE must be [A-Za-z0-9._-]+ (e.g. \$(date +%F)): $DATE"
if [ -n "$FILES" ]; then
    FILES=$(cd "$(dirname "$FILES")" && pwd)/$(basename "$FILES")
    [ -f "$FILES" ] || die "no such list file: $FILES"
fi
[ -d "$ROOT" ] || die "no such directory: $ROOT"
ROOT=$(cd "$ROOT" && pwd)
BASE=$ROOT/cormorant_repro_$DATE
CLONE=$BASE/cormorant
[ -e "$CLONE" ] && die "$CLONE exists — pick another DATE or remove it first"

# --- 0. disk --------------------------------------------------------------
echo "== disk"
df -h "$ROOT" / | sed 's/^/   /'
avail_gb=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc 0-9)
[ "$avail_gb" -ge "$MIN_FREE_GB" ] || die "$ROOT has ${avail_gb} GB free, the reproduction needs >= ${MIN_FREE_GB} GB"
root_gb=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
[ "$root_gb" -ge 3 ] || echo "   warning: / has ${root_gb} GB free — every shell must source $BASE/env.sh (TMPDIR etc.)"

# --- 1. clone exactly as README Quick start §1 ----------------------------
mkdir -p "$BASE"/{tmp,cache/pip,cache/hf,logs}
cat > "$BASE/env.sh" <<EOF
# source this in every shell of the reproduction (/ is nearly full)
export TMPDIR=$BASE/tmp
export XDG_CACHE_HOME=$BASE/cache
export PIP_CACHE_DIR=$BASE/cache/pip
export HF_HOME=$BASE/cache/hf
# the host pip.conf adds an unreachable extra index (3-9 min per venv); host-specific, not a doc gap
export PIP_CONFIG_FILE=/dev/null
EOF
export TMPDIR=$BASE/tmp

echo "== clone $URL -> $CLONE"
t0=$(date +%s)
git clone -q "$URL" "$CLONE"
git -C "$CLONE" submodule update --init -q
echo "   $(( $(date +%s) - t0 )) s, $(du -sh "$CLONE" | cut -f1) with submodules"

# --- 2. overlay the unpushed work -----------------------------------------
BASE_SHA=$(git -C "$CLONE" rev-parse HEAD)
MAIN_SHA=$(git -C "$MAIN" rev-parse HEAD)
git -C "$MAIN" cat-file -e "$BASE_SHA^{commit}" 2>/dev/null ||
    die "origin's HEAD $BASE_SHA is not in $MAIN — main is behind origin; pull first (remove $BASE, re-run)"
if ! git -C "$MAIN" merge-base --is-ancestor "$BASE_SHA" "$MAIN_SHA"; then
    git -C "$MAIN" merge-base --is-ancestor "$MAIN_SHA" "$BASE_SHA" &&
        die "$MAIN HEAD is behind origin's HEAD ${BASE_SHA:0:7} — pull first (remove $BASE, re-run)"
    die "$MAIN HEAD and origin's HEAD ${BASE_SHA:0:7} have diverged — rebase first (remove $BASE, re-run)"
fi
ahead=$(git -C "$MAIN" rev-list --count "$BASE_SHA..$MAIN_SHA")
[ "$ahead" = 0 ] || echo "   main is $ahead commit(s) ahead of origin — included in the overlay"

# The submodules (hw/cormorant_hw_128, hw/cormorant_test_stand) stay at the
# pushed commits: Vivado builds dirty them and their unpushed work cannot be
# overlaid here.  hw/test_data (tracked in the main repo) IS overlaid.
mapfile -t SUBS < <(git -C "$MAIN" config -f .gitmodules --get-regexp '\.path$' 2>/dev/null | awk '{print $2}')
EXCL=()
for s in "${SUBS[@]}"; do EXCL+=(":(exclude)$s"); done
git -C "$MAIN" diff --no-color --no-ext-diff "$BASE_SHA" --binary -- . "${EXCL[@]}" > "$BASE/overlay.patch"
if [ "${#SUBS[@]}" -gt 0 ]; then
    sub_diff=$(git -C "$MAIN" diff "$BASE_SHA" --stat -- "${SUBS[@]}" 2>/dev/null || true)
    [ -z "$sub_diff" ] || { echo "   note: submodule changes NOT overlaid (the clone uses the pushed hw):"; echo "$sub_diff" | sed 's/^/     /'; }
fi

GIT_ID=(-c user.name=repro -c user.email=repro@localhost -c commit.gpgsign=false)
if [ -s "$BASE/overlay.patch" ]; then
    git -C "$CLONE" apply --index --whitespace=nowarn "$BASE/overlay.patch"
    git -C "$CLONE" "${GIT_ID[@]}" commit -q -m "local: unpushed working-copy changes (not for push)"
    echo "   overlay: $(git -C "$CLONE" show --stat --format= HEAD | tail -1)"
else
    echo "   overlay: no tracked changes"
fi

: > "$BASE/overlay_files.txt"
if [ -n "$FILES" ]; then
    while IFS= read -r line || [ -n "$line" ]; do
        p=${line%%#*}; p=$(echo "$p" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')
        [ -n "$p" ] || continue
        [ -e "$MAIN/$p" ] || die "listed path not in $MAIN: $p"
        n=0
        while IFS= read -r -d '' f; do
            mkdir -p "$CLONE/$(dirname "$f")"
            cp -p "$MAIN/$f" "$CLONE/$f"
            printf '%s\n' "$f" >> "$BASE/overlay_files.txt"
            n=$((n + 1))
        done < <(git -C "$MAIN" ls-files --others --exclude-standard -z -- "$p")
        if [ "$n" = 0 ]; then
            insub=
            for s in "${SUBS[@]}"; do case "$p/" in "$s"/*) insub=$s ;; esac; done
            if [ -n "$insub" ]; then
                echo "   skip $p: inside submodule $insub — not overlaid"
            elif [ -n "$(git -C "$MAIN" ls-files -- "$p")" ]; then
                echo "   skip $p: tracked (already in overlay.patch)"
            else
                echo "   skip $p: ignored by .gitignore — a fresh clone lacks it too (a finding if the build needs it)"
            fi
        fi
    done < "$FILES"
fi
if [ -s "$BASE/overlay_files.txt" ]; then
    (cd "$CLONE" && tr '\n' '\0' < "$BASE/overlay_files.txt" | xargs -0 git add --)
    git -C "$CLONE" "${GIT_ID[@]}" commit -q -m "local: unpushed new files"
    echo "   new files: $(wc -l < "$BASE/overlay_files.txt") copied"
fi

# never push from the clone
git -C "$CLONE" remote set-url --push origin DISABLED-repro-clone-never-push
git -C "$CLONE" submodule foreach -q 'git remote set-url --push origin DISABLED-repro-clone-never-push'

{
    echo "date:        $DATE ($(date -Is))"
    echo "main:        $MAIN @ $MAIN_SHA ($ahead ahead of origin)"
    echo "url:         $URL"
    echo "origin HEAD: $BASE_SHA"
    echo "clone HEAD:  $(git -C "$CLONE" rev-parse HEAD)"
    git -C "$CLONE" log --format='  %h %s' "$BASE_SHA..HEAD"
    echo "submodules:"; git -C "$CLONE" submodule status | sed 's/^/  /'
    echo "patch:       $(wc -c < "$BASE/overlay.patch") bytes, $(grep -c '^diff --git' "$BASE/overlay.patch" || true) files"
    echo "new files:   $(wc -l < "$BASE/overlay_files.txt")"
    echo "clone size:  $(du -sh "$CLONE" | cut -f1)"
} > "$BASE/clone_info.txt"

# --- the agent brief and the GAPS.md skeleton, placeholders filled ---------
TPL=$(cd "$(dirname "$0")/../templates" && pwd)
fill() {
    sed -e "s|{{BASE}}|$BASE|g; s|{{CLONE}}|$CLONE|g; s|{{DATE}}|$DATE|g" \
        -e "s|{{CLONE_HEAD}}|$(git -C "$CLONE" rev-parse --short HEAD)|g" \
        -e "s|{{ORIGIN_HEAD}}|${BASE_SHA:0:7}|g" \
        -e "s|{{BOARD}}|${BOARD_HOST:-192.168.100.8}|g" \
        -e "s|{{KEY}}|${BOARD_KEY:-~/.ssh/kv260-testkey}|g" \
        -e "s|{{XILINX}}|${XILINX_ROOT:-/mnt/data/xilinx/2025.2}|g" "$1"
}
fill "$TPL/new_user_brief.md" > "$BASE/brief.md"
fill "$TPL/GAPS_template.md" > "$BASE/GAPS.md"
if grep -n '{{' "$BASE/brief.md" "$BASE/GAPS.md"; then die "unfilled placeholders above"; fi

echo "== done: $BASE"
sed 's/^/   /' "$BASE/clone_info.txt"
echo "   brief: $BASE/brief.md   GAPS skeleton: $BASE/GAPS.md"
echo "   next: board_snapshot.sh $BASE/board_before.txt; launch the new-user agent with brief.md (SKILL.md §B4)"
