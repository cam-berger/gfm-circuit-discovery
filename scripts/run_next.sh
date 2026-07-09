cd ~/Research/projects/gfm-circuit-discovery
echo "=============== sense-codon control  $(date +%H:%M:%S) ==============="
conda run --no-capture-output -n gfm-thrust1 python scripts/train_uniform.py --sense-only --steps 8000 || echo "SENSE FAILED"
for C in B C D; do
  echo "=============== cell $C  $(date +%H:%M:%S) ==============="
  conda run --no-capture-output -n gfm-thrust1 python scripts/train_cell.py --cell $C --steps 8000 || echo "CELL $C FAILED"
done
echo "ALL RUNS DONE $(date +%H:%M:%S)"
