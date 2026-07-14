# Study template: mechanistic discovery in a real GFM

Copy this directory to `studies/<model>-<computation>/` before broad internal screening.

## Study identity

- Model/checkpoint:
- Tokenizer and position scheme:
- Pretraining objective and corpus scope:
- Biological computation:
- Model-native behavioral metric:
- Primary dataset and held-out/sealed dataset:
- Compute budget:

## Required documents

- `PREREGISTRATION.md` — confirmatory gates, controls, thresholds, and stopping rules.
- `CLAIMS.md` — claim ledger with evidence level, status, scope, and next bridge.
- `DECISIONS.md` — dated deviations and decisions made after observing data.
- `RESULTS.md` — human-readable index into immutable result artifacts and manifests.

## Stage gates

1. **Behavior:** the target checkpoint demonstrably performs the computation.
2. **Instrument:** positive/negative controls, sensitivity, resolution, and invariants pass.
3. **Localization:** observational screens produce a frozen candidate set.
4. **Necessity:** controlled intervention changes the predeclared behavior.
5. **Mediation/path:** rescue, patching, or edge/path intervention supports a directed mechanism.
6. **Replication:** the strongest claim survives a held-out or sealed test.
7. **Interpretation:** biological origin/function is stated only if its independent bridge is crossed.

A study may stop at any gate. A well-powered null or diagnosed instrument limit is a valid result.

## Code and artifacts

- Add architecture access under `src/gfmcd/adapters/`.
- Keep biological assay logic in a study-specific package.
- Put run scripts under `scripts/<study>/`.
- Put immutable results and manifests under `results/<study>/`.
- Generate figures from recorded artifacts under `figures/<study>/`.
