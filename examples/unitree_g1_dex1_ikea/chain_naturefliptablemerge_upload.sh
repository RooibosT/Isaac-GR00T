#!/usr/bin/env bash
# Upload checkpoint-30000 of the two-session nature flip run (nature_fliptable + nature_fliptable_new).
#
# Not the arm8 argmin (22000): from 26,000 on arm8 sits within 1% of it, and among those
# checkpoints 30000 has the lowest EE8. The per-session scans settle it -- 22000 is worse than
# both specialists on grip, 30000 is better than both on every metric. The reasoning and the
# per-session table go in --extra-note (datasets/naturefliptablemerge_extra_note.md, generated
# from scan.json and outputs/<exp>/per_session_*.json). Set CHECKPOINT=<step> to override.
set -uo pipefail
ROOT=/root/01_IKEA/Isaac-GR00T
HUB=/root/02_hub/datasets
B=g1_dex1_ikea_relarm_3view_aug_b64_naturefliptablemerge_absarm
REPO=${REPO:-RooibosT/gr00t-n1.7-g1-dex1-naturefliptablemerge-absarm-30hz-h40}
CHAIN_LOG=$ROOT/datasets/chain_nature_fliptable_merge_absarm.log
LOG=$ROOT/datasets/upload_naturefliptablemerge.log
NOTE=$ROOT/datasets/naturefliptablemerge_extra_note.md
CKPT=${CHECKPOINT:-30000}
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

grep -q "chain complete" "$CHAIN_LOG" 2>/dev/null || { say "the training chain has not completed"; exit 1; }
[ -s "$NOTE" ] || { say "no extra note at $NOTE"; exit 1; }
source "$ROOT/.venv/bin/activate"

say "uploading checkpoint-$CKPT of $B -> $REPO"
python "$HUB/upload_best_ckpt.py" \
    --exp "$B" \
    --repo "$REPO" \
    --config g1_dex1_ikea_absarm_3view_aug_config.py \
    --dataset-repo "RooibosT/nature_fliptable + nature_fliptable_new (merged locally -- no single hub id holds this split)" \
    --train-eps 204 --train-frames 146163 --val-eps 20 --val-frames 15821 \
    --max-steps 34000 --save-steps 2000 --scan-stride 4 \
    --task-title "IKEA flip the table, both nature-room sessions" \
    --data-note "$(cat "$HUB/naturefliptablemerge_data_note.txt")" \
    --state-dim 46 \
    --state-desc "legs 12, waist 3, both arms 7+7, both grippers 1+1, base_gravity 3, both FK EEF 6+6." \
    --extra-note "$(cat "$NOTE")" \
    --checkpoint "$CKPT" ${DRY_RUN:+--dry-run} \
    >> "$LOG" 2>&1 && say "upload done" || say "UPLOAD FAILED (see $LOG)"
