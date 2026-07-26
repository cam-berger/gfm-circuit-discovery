#!/usr/bin/env bash
# Re-fetch and hash-verify every external input for the DNABERT-2 core-promoter study.
# `data/external/` is gitignored; this script plus studies/dnabert2_core_promoter/DATA_ACQUISITION.md
# are the tracked record. Idempotent: existing files are verified, not re-downloaded.
#
#   bash scripts/dnabert2/fetch_data.sh        # from repo root
set -euo pipefail

DEST="data/external/epdnew"
EPD="https://epd.expasy.org/ftp/epdnew/H_sapiens/current"
JASPAR="https://jaspar.elixir.no/api/v1/matrix/MA0108.3/?format=json"
HG38="${HG38:-$HOME/Research/data/genomes/hg38.fa}"

mkdir -p "$DEST"

sha() { shasum -a 256 "$1" | awk '{print $1}'; }

get() {  # get <url> <path> <expected-sha256>
  local url="$1" path="$2" want="$3"
  if [[ ! -f "$path" ]]; then
    echo "fetching $(basename "$path")"
    curl -sSL --max-time 300 "$url" -o "$path"
  fi
  local got; got="$(sha "$path")"
  if [[ "$got" != "$want" ]]; then
    echo "FAIL $(basename "$path"): sha256 $got != pinned $want" >&2
    exit 1
  fi
  echo "  ok  $(basename "$path")  $want"
}

echo "== EPDnew human Rel. 006 (hg38) =="
get "$EPD/Hs_EPDnew.dat" "$DEST/Hs_EPDnew.dat" \
    88cfaf67ce2ad7a84ac5be779a54e9ded443e626137936526e37a839a06f0447
get "$EPD/Hs_EPDnew.bed" "$DEST/Hs_EPDnew.bed" \
    3ecac81b26ed99cc4d91e84f9dedd123de250a42981920d8d065e09c2ca114f8

echo "== JASPAR MA0108.3 (TBP) =="
# The JASPAR API is not byte-stable across responses, so the matrix content is asserted
# rather than the file hash: 7 positions, consensus TATAAAA.
if [[ ! -f "$DEST/JASPAR_MA0108.3_TBP.json" ]]; then
  curl -sSL --max-time 120 "$JASPAR" -o "$DEST/JASPAR_MA0108.3_TBP.json"
fi
python3 - "$DEST/JASPAR_MA0108.3_TBP.json" <<'PY'
import json, sys
pfm = json.load(open(sys.argv[1]))["pfm"]
assert len(pfm["A"]) == 7, f'expected 7 positions, got {len(pfm["A"])}'
cons = "".join(max("ACGT", key=lambda b: pfm[b][j]) for j in range(7))
assert cons == "TATAAAA", f"unexpected consensus {cons}"
print(f"  ok  MA0108.3  L=7  consensus={cons}")
PY

echo "== reference genome =="
if [[ ! -f "$HG38" ]]; then
  echo "FAIL hg38 not found at $HG38 (set HG38=...)" >&2; exit 1
fi
[[ -f "$HG38.fai" ]] || { echo "FAIL missing $HG38.fai" >&2; exit 1; }
echo "  ok  $HG38 (md5 pinned in DATA_ACQUISITION.md; not re-hashed here, 3.2 GB)"

echo "== model checkpoint (local HF cache; never downloaded at run time) =="
SNAP="$HOME/.cache/huggingface/hub/models--zhihan1996--DNABERT-2-117M/snapshots/7bce263b15377fc15361f52cfab88f8b586abda0"
[[ -d "$SNAP" ]] || { echo "FAIL DNABERT-2 snapshot not in HF cache: $SNAP" >&2; exit 1; }
got="$(sha "$SNAP/pytorch_model.bin")"
want=7ff39ec77a484dd01070a41bfd6e95cdd7247bec80fe357ab43a4be33687aeba
[[ "$got" == "$want" ]] || { echo "FAIL checkpoint sha256 $got != $want" >&2; exit 1; }
echo "  ok  pytorch_model.bin  $want"

shasum -a 256 "$DEST/Hs_EPDnew.dat" "$DEST/Hs_EPDnew.bed" > "$DEST/EPDNEW.sha256"
echo "all inputs verified"
