cd ~/Research/projects/gfm-circuit-discovery
for LAM in 0.75 1.0 1.5 2.0; do
  echo "=============== lambda $LAM  $(date +%H:%M:%S) ==============="
  conda run --no-capture-output -n gfm-thrust1 python scripts/train_uniform.py --lam $LAM --steps 8000 || echo "ARM $LAM FAILED"
done
echo "SWEEP DONE $(date +%H:%M:%S)"
