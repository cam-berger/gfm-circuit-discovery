# Run the audited stage-two target on Colab

Use `run_atlas_stage2_colab_v3.ipynb` only to reproduce the original target execution. For the completed
run's numerical recovery, use `run_atlas_stage2_recovery_a3.ipynb` with the v6 overlay. The manual cells
below preserve the original pre-A3 contract and are not instructions to overwrite the completed run.

The A3 recovery is defined in `../MECHANISM_VALIDATION_AMENDMENT_A3.md`: it preserves the failed target
manifest, accepts no failure except the independently reconstructed cross-batch reference discrepancy in
`(1e-6,1e-4]`, and writes new `*_a3_v2` artifacts. The v6 overlay also supplies the post-audit DAG builder
and independent DAG validator; it contains no Stage 2 target outcome.

This overlay continues the completed `gfm_atlas_screen_colab_v1` environment. It contains the sealed
internal-validation cohort, audited stage-one and stage-two packages, stage-two scripts, and the local
256-packet `HEALTH_PASS` shards. It does not contain or imply a mechanism result.

## 1. Mount Drive, verify the overlay, and extract it

```python
from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path
import hashlib, json, shutil, zipfile

drive_root = Path('/content/drive/MyDrive/gfm_atlas_screen_v1_results')
overlay = drive_root / 'gfm_atlas_stage2_colab_overlay_v3.zip'
sidecar = drive_root / 'gfm_atlas_stage2_colab_overlay_v3.zip.sha256.json'
expected = json.loads(sidecar.read_text())['sha256']
h = hashlib.sha256()
with overlay.open('rb') as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b''):
        h.update(block)
assert h.hexdigest() == expected, (h.hexdigest(), expected)

repo = Path('/content/gfm_atlas_screen/repo')
persisted_repo = Path('/content/drive/MyDrive/gfm_atlas_screen/repo')
checkpoint = repo / 'data/models/splicebert_official_archive/extracted/models/SpliceBERT.1024nt/pytorch_model.bin'
if not checkpoint.is_file():
    # A fresh Colab runtime does not retain /content from the discovery run. Restore the
    # hash-verified repository that the discovery notebook persisted on Drive.
    assert (persisted_repo / 'BUNDLE_MANIFEST.json').is_file(), persisted_repo
    shutil.copytree(persisted_repo, repo, dirs_exist_ok=True)
assert checkpoint.is_file(), checkpoint
with zipfile.ZipFile(overlay) as archive:
    archive.extractall(repo)

stage2_out = Path('/content/drive/MyDrive/gfm_atlas_stage2_v1_results')
local_health = repo / 'results/splicebert_head_context_atlas/atlas_stage2_mechanism_v1'
stage2_out.mkdir(parents=True, exist_ok=True)
shutil.copytree(local_health, stage2_out, dirs_exist_ok=True)
assert (stage2_out / 'run_manifest.health.json').is_file()
```

Do not continue if the hash assertion fails.

## 2. Run only the untouched target phase on CUDA

```python
import os, subprocess

env = os.environ.copy()
env['PYTHONHASHSEED'] = '0'
env['TOKENIZERS_PARALLELISM'] = 'false'
subprocess.run([
    'python', 'scripts/splicebert_head_context_atlas/run_atlas_stage2_mechanisms.py',
    '--package', 'results/splicebert_head_context_atlas/atlas_stage2_package_v1.json',
    '--package-audit', 'results/splicebert_head_context_atlas/atlas_stage2_package_v1.audit.json',
    '--stage1-package', 'results/splicebert_head_context_atlas/atlas_mechanism_package_v1.json',
    '--output-dir', str(stage2_out),
    '--phase', 'target', '--health-n', '256', '--batch', '8', '--device', 'cuda',
], cwd=repo, env=env, check=True)
```

The target run is immutable and resumable. Re-running the same cell resumes under the same contract.

## 3. Audit, merge, analyze, and independently reconstruct

```python
merged = stage2_out / 'atlas_stage2_mechanism_v1_merged'
analysis = stage2_out / 'atlas_stage2_mechanism_analysis_v1'

commands = [
    ['python', 'scripts/splicebert_head_context_atlas/validate_atlas_stage2_mechanisms.py',
     '--run-dir', str(stage2_out),
     '--package', 'results/splicebert_head_context_atlas/atlas_stage2_package_v1.json',
     '--package-audit', 'results/splicebert_head_context_atlas/atlas_stage2_package_v1.audit.json'],
    ['python', 'scripts/splicebert_head_context_atlas/merge_atlas_stage2_mechanisms.py',
     '--run-dir', str(stage2_out),
     '--package', 'results/splicebert_head_context_atlas/atlas_stage2_package_v1.json',
     '--prefix', str(merged)],
    ['python', 'scripts/splicebert_head_context_atlas/analyze_atlas_stage2_mechanisms.py',
     '--merged-prefix', str(merged),
     '--package', 'results/splicebert_head_context_atlas/atlas_stage2_package_v1.json',
     '--package-audit', 'results/splicebert_head_context_atlas/atlas_stage2_package_v1.audit.json',
     '--execution-audit', str(stage2_out / 'validation_audit.json'),
     '--run-dir', str(stage2_out),
     '--prefix', str(analysis)],
    ['python', 'scripts/splicebert_head_context_atlas/validate_atlas_stage2_analysis.py',
     '--merged-prefix', str(merged),
     '--analysis-prefix', str(analysis),
     '--package', 'results/splicebert_head_context_atlas/atlas_stage2_package_v1.json',
     '--run-dir', str(stage2_out)],
]
for command in commands:
    subprocess.run(command, cwd=repo, env=env, check=True)
```

Do not read mechanism labels until the final analysis audit reports `PASS` with `n_failures=0`.
