#!/usr/bin/env bash
# Fetch the two large data files (GTF + hg38 FASTA) into data/genomes/ under the repo root.
# The SpliceBERT.1024nt checkpoint is BUNDLED in the zip (data/models/...), so it is NOT fetched here.
# Run from the repo root. Idempotent: skips files already present with the right size.
set -euo pipefail
cd "$(dirname "$0")/../../.."   # -> repo root (colab_package is 3 dirs deep under the repo)
mkdir -p data/genomes
cd data/genomes

GTF=gencode.v50.annotation.gtf.gz
GTF_SHA=83fba3e9b03f0b8c958f3595c6c350adc55f468abf8b0e47b6d5284cfe13a453
if [ ! -f "$GTF" ]; then
  echo "[fetch] comprehensive GENCODE v50 GTF (124 MB)"
  curl --fail -sSL -o "$GTF.part" \
    https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_50/gencode.v50.annotation.gtf.gz
  mv "$GTF.part" "$GTF"
fi
GOT=$(shasum -a 256 "$GTF" | cut -d' ' -f1)
[ "$GOT" = "$GTF_SHA" ] || { echo "GTF sha mismatch: $GOT != $GTF_SHA"; exit 1; }
echo "[fetch] GTF sha OK"

if [ ! -f hg38.fa ]; then
  echo "[fetch] hg38 FASTA (~3 GB uncompressed) from UCSC"
  curl --fail -sSL -o hg38.fa.gz https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
  gunzip -f hg38.fa.gz
fi
echo "[fetch] done. data/genomes/ ready:"
ls -lh
