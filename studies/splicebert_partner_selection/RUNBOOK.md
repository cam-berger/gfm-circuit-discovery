# Runbook

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

Run from repository root. Commands are also summarized in the root implementation document.

## 0. Tests

```bash
uv sync --extra splicebert --extra dev
uv run pytest -q tests/test_splicing_*.py
```

## 1. Resources

```bash
bash scripts/splicebert/fetch_resources.sh
```

Inspect extracted model directories and set `model.path` explicitly.

## 2. Smoke build

```bash
bash scripts/splicebert/run_smoke.sh
```

Review logs, manifests, and 20 hand-inspected examples before full construction.

## 3. Full data and baseline

```bash
uv run python scripts/splicebert/build_dataset.py
uv run python scripts/splicebert/evaluate_local_baselines.py
```

## 4. Behavior matrix

```bash
bash scripts/splicebert/run_validation_pipeline.sh
```

The script intentionally does not open test data.

## 5. Stop/go decision

Record the decision in `DECISIONS.md`. Trace only if the primary frozen condition passes the behavioral gate.

## 6. Discovery trace

```bash
uv run python scripts/splicebert/trace_heads.py --checkpoint <additive-frozen-checkpoint>
uv run python scripts/splicebert/patch_conflicts.py --checkpoint <checkpoint> --corruption donor-mask
```

Add partner-specific patching before a final circuit claim.

## 7. Test unblinding

Freeze preregistration, checkpoint, component set, scripts, and commit. Then run test commands with explicit flags.
