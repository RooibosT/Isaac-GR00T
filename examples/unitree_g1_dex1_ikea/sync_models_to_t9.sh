#!/usr/bin/env bash
# Mirror every finished model to the T9 drive in ONE rsync call.
#
#   DRY=1 bash sync_models_to_t9.sh     # list what would move, transfer nothing
#   bash sync_models_to_t9.sh           # transfer
#
# One call on purpose: with password auth that means exactly one prompt for the
# whole 600+ GB instead of one per directory. If you have set up
# `ssh-copy-id chan@143.248.94.3` there is no prompt at all and this can run
# under nohup.
#
# Skips any run whose training process is still alive, so a half-written
# checkpoint never reaches the drive. `--partial --append-verify` means an
# interrupted transfer resumes rather than restarts, so re-running is always safe
# and only moves what is missing or changed.
set -uo pipefail

ROOT="/home/chan/IKEA/Isaac-GR00T"
DEST_HOST="${DEST_HOST:-chan@143.248.94.3}"
DEST_DIR="${DEST_DIR:-/media/chan/T9/IKEA/hub/models}"
DRY="${DRY:-}"

cd "$ROOT"

running() {   # a run is in flight if launch_finetune still names its output_dir
    pgrep -af 'launch_finetune\.py' 2>/dev/null | grep -q -- "--output_dir[= ][^ ]*/$1\b"
}

# EXCLUDE holds space-separated run names to leave behind even though training
# finished -- e.g. a run whose checkpoints have not been scanned yet, so nobody
# knows which one to keep and shipping all of them would waste the drive.
EXCLUDE="${EXCLUDE:-}"
excluded() {
    for e in $EXCLUDE; do [ "$1" = "$e" ] && return 0; done
    return 1
}

LIST=$(mktemp); trap 'rm -f "$LIST"' EXIT
for d in outputs/g1_dex1_*; do
    n=$(basename "$d")
    if running "$n"; then echo "skip (still training): $n" >&2; continue; fi
    if excluded "$n"; then echo "skip (excluded): $n" >&2; continue; fi
    echo "$d" >> "$LIST"
done
for m in models/GR00T-N1.7-3B models/bct-relarm-aug-adopted models/Team-RAMEN; do
    [ -d "$m" ] && echo "$m" >> "$LIST"
done

echo "items: $(wc -l < "$LIST")"
du -shc $(cat "$LIST") 2>/dev/null | tail -1
echo "dest: $DEST_HOST:$DEST_DIR"
echo

# -r is NOT implied by -a when --files-from is used: rsync turns recursion off so
# a file list means exactly those entries. Without it this copies 29 empty dirs.
RS=(rsync -aHr --info=progress2,stats2 --partial --append-verify
    --exclude '*.lock' --exclude '__pycache__/'
    --files-from="$LIST" --relative
    -e 'ssh -o ConnectTimeout=15')
[ -n "$DRY" ] && RS+=(--dry-run)

# --relative keeps the outputs/ and models/ prefixes, so the tree lands as
# $DEST_DIR/outputs/<run> and $DEST_DIR/models/<base model>
"${RS[@]}" . "$DEST_HOST:$DEST_DIR/"
rc=$?
echo "rsync exit $rc"
exit "$rc"
