# Phase 2C.3 reviewer questions

No answer, reviewer identity, signature, or approval is prefilled. Answers must identify
the exact reviewed artifact hashes; a matching hash proves identity, not scientific
approval.

## Traffic engineering

1. Do the proposed E1 boundaries measure junction-specific approach entries, and are
   shared/adjacent boundaries either disjoint or reconciled without double counting?
2. Are detector lanes and positions suitable under lane changes, teleport entry, cyclic
   re-entry, and short-edge traversal? Which cases require fail-closed exclusion?
3. For each frozen TLS, are green-link inventories complete and are commanded yellow,
   red-hold, minimum-green, and maximum-green transitions physically acceptable?
4. Is TLS `96621100`'s native all-red → all-yellow → all-green → all-red sequence
   intentional? What clearance interpretation supports it?
5. Which two-TLS competing-phase subset supports conventional max-pressure, and how
   should the remaining gate/red-hold controllers be named?

## Forecast methodology

6. Is the seven-feature local schema and `(t,t+30]` target scientifically appropriate?
7. Which count likelihood/output constraint prevents negative predictions?
8. Are the five demand partitions and overlapping temporal windows leakage-safe at the
   demand-realization level?
9. Are zero, persistence, historical-mean, and moving-average baselines sufficient, and
   what metric/selection rule is primary?

## RL and experimental design

10. Are five independent training seeds and ten paired held-out demand seeds per profile
    and controller adequate for the proposed estimand?
11. Is training seed correctly treated as the independent learned-policy unit, and is
    the hierarchical paired analysis/multiplicity plan acceptable?
12. What fixed training budget and validation-only checkpoint rule should be frozen?
13. Are reward reference values and their calibration quantiles scientifically justified
    and isolated from validation/test/evaluation demand?
14. Is the 60-second common learned-policy warm-up acceptable, and are native baseline
    comparisons correctly labeled as full-operation secondary comparisons?

## Outcomes and claims

15. Does demand-normalized censored waiting appropriately prevent insertion blocking
    from appearing beneficial, with insertion/completion/backlog outcomes mandatory?
16. Is the failure/teleport/collision handling rule specified before results are seen?
17. Should all claims remain limited to an uncalibrated Tahrir-derived simulation until
    independent field counts, queues, travel times, saturation flow, and validation
    targets exist?
18. What evidence, if any, would be required before a Taksim transfer claim is examined?

