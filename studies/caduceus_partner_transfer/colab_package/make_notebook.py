"""Emit caduceus_t1_colab.ipynb — a thin, smoke-first Colab wrapper around the validated .py modules.
Keeps all tested logic in caduceus_scorer/caduceus_t1/run_t1 (not duplicated in cells). Run: python make_notebook.py
"""
import json

def md(*s): return {"cell_type": "markdown", "metadata": {}, "source": list(s)}
def code(*s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": list(s)}

cells = [
    md("# Caduceus-PS T1 — behavioural transfer (CUDA)\n",
       "Runs the **frozen prereg v2** assay: does the partner-specific donor↔acceptor coupling behaviour "
       "transfer to a BiMamba SSM?  Requires a **GPU runtime** (Runtime → Change runtime type → GPU).\n\n",
       "**Discipline (don't waste GPU):** all constructs, controls, and statistics are pre-built/validated "
       "locally (`build_cohort.py`, mock dry-runs); this notebook only (1) loads Caduceus, (2) runs a "
       "**smoke gate** that HARD-STOPS on any memorial failure, then (3) does one clean forward pass + the "
       "frozen §6 decision.  Coordinate correctness (prefix=0 / trailing [SEP] / no-padding) is asserted at "
       "scorer init and re-checked here (E2/E3/E6)."),
    md("### 1 · Install (CUDA kernels — no CPU/MPS fallback)"),
    code("!pip -q install 'mamba-ssm>=2.2' 'causal-conv1d>=1.4' 'transformers>=4.44' scipy einops\n",
         "import torch; assert torch.cuda.is_available(), 'Enable a GPU runtime (Runtime>Change runtime type>GPU).'\n",
         "print('CUDA:', torch.cuda.get_device_name(0))"),
    md("### 2 · Upload the package\n",
       "Upload `caduceus_t1_colab.zip` (this package) and `cohort_constructs.jsonl` (built locally by "
       "`build_cohort.py`), or mount Drive. The four .py modules must be importable from the cwd."),
    code("import os, glob\n",
         "for f in ['caduceus_scorer.py','caduceus_t1.py','run_t1.py','cohort_constructs.jsonl']:\n",
         "    assert os.path.exists(f), f'missing {f} — upload the package + cohort'\n",
         "print('files present:', [os.path.basename(f) for f in glob.glob('*.py')])"),
    md("### 3 · SMOKE GATE (hard-stop) — load Caduceus + memorials on a tiny subset\n",
       "Validates on the REAL model what the mock could not: coordinate roundtrip, `mlm_health>0`, "
       "`A_donor>0`.  STOPS before the full run if any fails (no wasted GPU)."),
    code("import json, numpy as np\n",
         "from transformers import AutoTokenizer, AutoModelForMaskedLM\n",
         "from caduceus_scorer import CaduceusScorer\n",
         "MID='kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16'\n",
         "tok=AutoTokenizer.from_pretrained(MID, trust_remote_code=True)\n",
         "model=AutoModelForMaskedLM.from_pretrained(MID, trust_remote_code=True).cuda().eval()\n",
         "sc=CaduceusScorer(model, tok, device='cuda', batch_size=32)   # __init__ asserts prefix=0/trailing[SEP]/ids\n",
         "rows=[json.loads(l) for l in open('cohort_constructs.jsonl') if l.strip()]\n",
         "sub=rows[:64]\n",
         "# coordinate roundtrip + coordinate-sensitivity on a known donor\n",
         "r0=sub[0]; seq,d=r0['constructs']['clean']\n",
         "m_d=sc.donor_margin([seq],[d])[0]; m_off=sc.donor_margin([seq],[d+1])[0]\n",
         "# memorials\n",
         "mlm=sc.mlm_health([r['constructs']['clean'][0] for r in sub[:20]])\n",
         "clean=sc.donor_margin([r['constructs']['clean'][0] for r in sub],[r['constructs']['clean'][1] for r in sub])\n",
         "dead=sc.donor_margin([r['constructs']['donor_dead'][0] for r in sub],[r['constructs']['donor_dead'][1] for r in sub])\n",
         "A=float(np.mean(clean-dead))\n",
         "print(f'margin_donor(clean)={m_d:.3f}  margin@d+1={m_off:.3f}  mlm_health={mlm:.3f}  A_donor={A:.3f}')\n",
         "assert m_d>0, 'donor not recognised'; assert m_d>m_off, 'coordinate off-by-one (readout not at donor)'\n",
         "assert mlm>0, 'MLM health failed (not a working LM at these coords)'\n",
         "assert A>0, 'A_donor<=0 — donor-destruction anchor failed'\n",
         "print('SMOKE GATE PASS ✓ — proceed to the confirmatory run')"),
    md("### 4 · Confirmatory run (one clean pass) + frozen §6 decision"),
    code("import sys, run_t1\n",
         "sys.argv=['run_t1.py','--constructs','cohort_constructs.jsonl','--batch','32','--out','results_t1.json']\n",
         "run_t1.main()"),
    md("### 5 · Result\n",
       "`VERDICT ∈ {QUALIFIES-FOR-PHASE-3, DOES-NOT-QUALIFY (honest architecture null), INSTRUMENT-RANGE-LIMIT, "
       "UNDER-POWERED}`.  Download `results_t1.json`; report per the frozen prereg §6."),
    code("import json; print(json.dumps(json.load(open('results_t1.json')), indent=1)[:2000])\n",
         "from google.colab import files; files.download('results_t1.json')"),
]

nb = {"cells": cells, "metadata": {"accelerator": "GPU",
      "kernelspec": {"display_name": "Python 3", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open("caduceus_t1_colab.ipynb", "w"), indent=1)
print("wrote caduceus_t1_colab.ipynb")
