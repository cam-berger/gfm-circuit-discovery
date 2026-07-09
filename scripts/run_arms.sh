set -e
cd ~/Research/projects/gfm-circuit-discovery
for OBJ in ar mlm; do
  echo "=============== $OBJ  $(date +%H:%M:%S) ==============="
  conda run --no-capture-output -n gfm-thrust1 python scripts/train_nano.py --objective $OBJ --steps 8000
done
echo "ALL DONE $(date +%H:%M:%S)"
