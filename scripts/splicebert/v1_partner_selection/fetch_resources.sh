#!/usr/bin/env bash
set -euo pipefail

# Fetch only public upstream artifacts. The script never commits them; it writes
# SHA-256 manifests so experiments can pin the exact bytes used.
#
# Usage:
#   scripts/splicebert/fetch_resources.sh [gencode-release]
#
# Default release is 50 (GRCh38.p14, current when this plan was written in July
# 2026). Pass an explicit release to reproduce an older preregistration.

RELEASE="${1:-50}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXT="$ROOT/data/external/gencode/v${RELEASE}"
MODEL_ARCHIVE_DIR="$ROOT/data/models/splicebert_official_archive"
MODEL_TGZ="$MODEL_ARCHIVE_DIR/models.tar.gz"
mkdir -p "$EXT" "$MODEL_ARCHIVE_DIR"

fetch() {
  local url="$1"
  local out="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 --continue-at - "$url" -o "$out"
  elif command -v wget >/dev/null 2>&1; then
    wget -c "$url" -O "$out"
  else
    echo "Need curl or wget" >&2
    exit 2
  fi
}

GTF="$EXT/gencode.v${RELEASE}.basic.annotation.gtf.gz"
FASTA_GZ="$EXT/GRCh38.primary_assembly.genome.fa.gz"
FASTA="$EXT/GRCh38.primary_assembly.genome.fa"
fetch "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_${RELEASE}/gencode.v${RELEASE}.basic.annotation.gtf.gz" "$GTF"
fetch "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_${RELEASE}/GRCh38.primary_assembly.genome.fa.gz" "$FASTA_GZ"
if [[ ! -f "$FASTA" ]]; then
  gzip -dc "$FASTA_GZ" > "$FASTA"
fi

# Official SpliceBERT weights from the authors' Zenodo record. This archive may
# contain several checkpoints. We deliberately do not guess which extracted
# directory is the full model; inspect the candidates printed below and set
# model.path in configs/splicebert_partner_selection/base.json.
fetch "https://zenodo.org/record/7995778/files/models.tar.gz?download=1" "$MODEL_TGZ"
mkdir -p "$MODEL_ARCHIVE_DIR/extracted"
tar -xzf "$MODEL_TGZ" -C "$MODEL_ARCHIVE_DIR/extracted"

MANIFEST="$ROOT/data/external/splicebert_resources_v${RELEASE}.sha256"
{
  sha256sum "$GTF"
  sha256sum "$FASTA_GZ"
  sha256sum "$FASTA"
  sha256sum "$MODEL_TGZ"
} > "$MANIFEST"

echo
printf 'Downloaded resources. SHA-256 manifest: %s\n' "$MANIFEST"
echo "Candidate SpliceBERT model directories (look for config.json):"
find "$MODEL_ARCHIVE_DIR/extracted" -name config.json -print | sed 's#/config.json$##'
echo
echo "Next: set model.path in configs/splicebert_partner_selection/base.json."
