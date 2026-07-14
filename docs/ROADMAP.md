# Research roadmap

## North star

Build a transferable, evidence-gated practice for causal mechanistic discovery in genomic foundation models, then use cross-model comparisons to identify recurring computational motifs and architecture- or objective-specific solutions.

## Milestone 1 — method-first repository

**Status: in progress.**

- Make the transferable methodology the repository front door.
- Keep SpliceBERT as the worked example and detailed provenance record.
- Remove GFM-nano, reading-frame, repeat-copy, and retracted v1 implementations from the active tree.
- Maintain one evidence ladder and one agent/change protocol.

Exit condition: a new contributor can explain the workflow, current evidence, and repository map without reconstructing the project from experiment directories.

## Milestone 2 — freeze the portable protocol

- Turn the SpliceBERT preflight, provenance, intervention, and claim-tracking patterns into reusable interfaces.
- Identify which current utilities are model-generic and which remain SpliceBERT-specific.
- Add small contract tests for adapters, hook identity, intervention invariants, and result manifests.
- Define a minimal compute budget for behavior, screening, causal testing, and sealed replication.

Exit condition: the study template and adapter contract are sufficient to start a second model without copying a SpliceBERT experiment wholesale.

## Milestone 3 — select the next GFM

- Evaluate candidate models against the criteria in `TRANSFER_GUIDE.md`.
- Choose one biological computation with a native behavioral signal and strong counterfactual controls.
- Write the study preregistration before broad internal screening.

Exit condition: model, checkpoint, computation, assay, controls, stopping rules, and resource budget are frozen.

## Milestone 4 — execute the second worked case

- Qualify behavior and instrument.
- Localize candidates exploratorily.
- Run necessity, mediation, information, and path tests.
- Seal and replicate the strongest claim.
- Record nulls and retractions with the same prominence as positive findings.

Exit condition: the study reaches a justified point on the evidence ladder and states the first missing bridge.

## Milestone 5 — cross-model synthesis

- Compare algorithms rather than layer/head identities.
- Identify recurring motifs such as content addressing, positional clocks, routing bottlenecks, or distributed redundancy.
- Separate effects of architecture, pretraining objective, corpus, context length, and position representation.
- Revise the methodology based on where transfer succeeded or failed.

## Open SpliceBERT work

The ≈10.63 carrier has reached post-softmax attention but not OV/mediation or behavior. Its next legitimate questions are downstream write, causal use, and origin. These should not block transfer to another GFM; they are a separate live branch of the research program.
