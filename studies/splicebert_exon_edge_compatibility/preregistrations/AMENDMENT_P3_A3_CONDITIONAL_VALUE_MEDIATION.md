# Amendment P3-A3 — conditional value mediation

> **Tier: T2 — prospective provenance. Frozen on 2026-08-04 after the
> outcome-blind method audit and mechanical-only smoke v1, and before smoke v2,
> development preflight, or any target execution from the direct-edge P3
> family.**

This amendment corrects one identification gap in
`PREREGISTRATION_DIRECT_EDGE_P3_RELATIONAL_MEDIATION.md` and supersedes the A/V
endpoint-family and value-classification clauses in
`AMENDMENT_P3_A2_ARTIFACT_AND_CHANNEL_BINDINGS.md`. It does not change the
biological hypothesis, carrier intervention, route construction, cohorts,
sample sizes, bootstrap, thresholds, or exposure chronology. No `G`, `S_r`,
`Omega`, mediation, local Q/K, preflight-effect, or target-effect outcome from
this intervention family had been produced when the correction was frozen.

## 1. Identification problem

The four frozen A/V cells remain `A_R V_R` (`RR`), `A_T V_R` (`TR`),
`A_R V_T` (`RT`), and `A_T V_T` (`TT`). The parent protocol bounded value
mediation with `Omega(RT)-Omega(RR)`. That contrast measures the value change
conditional on reference routing. It does not bound the value change
conditional on target routing:

`Omega(TT)-Omega(TR)`.

The two differ by the A/V interaction. Therefore the reference-context value
contrast cannot by itself license a routing-specific classification.

## 2. Corrected estimands and simultaneous family

The total and routing contrasts are unchanged:

- `D_total = Omega(TT)-Omega(RR)`;
- `D_route = Omega(TR)-Omega(RR)`.

The value contrasts are now named by their routing context:

- `D_value_reference = Omega(RT)-Omega(RR)`;
- `D_value_target = Omega(TT)-Omega(TR)`;
- `interaction = D_value_target-D_value_reference`
  `= Omega(TT)-Omega(TR)-Omega(RT)+Omega(RR)`.

The max-absolute-studentized A/V mediation family is frozen as exactly
`{D_total,D_route,D_value_reference,D_value_target,interaction}`. The three
per-register routing contrasts `{r_0,r_1,r_2}` remain their own simultaneous
family. No legacy `D_value` alias is serialized; the old name may not authorize
a gate or mechanism classification.

## 3. Preflight sensitivity and resolution

Neither conditional value contrast has a point-direction gate. Development
preflight requires projected target `MDE80<=0.05` for both
`D_value_reference` and `D_value_target`, in addition to the existing carrier
`Omega`, `D_total`, `D_route`, and local-Q/K `Omega` requirements. The frozen
program therefore has six projected-resolution endpoints. `interaction` is
reported but has no separate MDE gate.

All projections retain 20,000 centered chromosome-by-strand whole-block
bootstrap draws, seed `20260806`, and fixed target `N=384`; observed effect
magnitudes do not enter the resolution calculation.

## 4. Target classification correction

Carrier/preflight instrument status and route-recorder status are evaluated at
different evidence levels. A failed carrier/preflight instrument remains
`INSTRUMENT_UNRESOLVED`. If the carrier instrument and positive carrier gates
pass but an A/V round-trip, target-Q/K no-op, or matched-head no-op recorder
fails, the established carrier result is preserved and the classification is
`P3_RELATIONAL_CARRIER_ROUTE_UNRESOLVED`, with the recorder failure named. A
route recorder may not retroactively erase carrier evidence.

The remaining route, Q/K, matched-head, and provenance gates are unchanged.
Conditional value evidence is classified as follows:

1. `P3_RELATIONAL_ROUTING_SPECIFIC` requires both conditional value
   simultaneous intervals wholly inside `[-0.05,+0.05]` and both conditional
   value `MDE80` values at most `0.05`.
2. `P3_RELATIONAL_MIXED_AV` requires at least one conditional value contrast
   with `MDE80<=0.05` whose simultaneous interval excludes zero.
3. If neither rule is met, the value contribution is unresolved and the final
   classification is `INCONCLUSIVE`, even when the localized routing route is
   detected.

The old single-`D_value` gate is forbidden from authorizing any of these rows.

## 5. Claim boundary

A routing-specific result may state that **the conditional value-mediated
contribution to `Omega` is bounded in both frozen routing contexts for this
intervention family**. It does not state that L3H1 value-vector changes are
themselves bounded, and it does not establish globally absent value mediation,
unchanged payload semantics, or a value-free model computation. All checkpoint,
behavior, sequence-context, intervention-family, and biological-interpretation
guardrails from the parent protocol and A2 remain in force.
