set -e
cd ~/Research/projects/gfm-circuit-discovery
for OBJ in ar mlm; do
  echo "=============== $OBJ mixed  $(date +%H:%M:%S) ==============="
  conda run --no-capture-output -n gfm-thrust1 python scripts/train_nano.py --objective $OBJ --pool mixed --steps 8000
done
echo "MIXED DONE $(date +%H:%M:%S)"
