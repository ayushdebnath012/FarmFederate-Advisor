# Tea VLM Federated-Adaptation Sweep

- Generated: 2026-07-27T17:29:32.151404+00:00
- Evidence type: validation-only sensitivity analysis
- Initialization: validation-selected centralized v6 checkpoint
- Aggregation: full-participation, sample-weighted FedAvg
- Client split: per-class Dirichlet label skew
- Frozen component: ResNet-50 visual backbone
- Locked test: not used for selection or plotted sweep values

## Source-group audit

- split seed: `42`
- train crops: `222`
- validation crops: `74`
- locked test crops not used: `75`
- train source images: `122`
- validation source images: `40`
- locked test source images not used: `38`
- pairwise source overlap: `0`
- train pairing coverage: `1.0`
- validation pairing coverage: `1.0`
- masked target shortcut tokens: `178`
- checkpoint sha256: `76abbd249a284d3be442fe2c77e9b38da6045fada095676c1bccd0acb0f9ac46`

## Validation macro-F1

| Clients | Final mean | Best mean | Mean label TV |
|---:|---:|---:|---:|
| 2 | 0.8924 | 0.9099 | 0.139 |
| 3 | 0.9046 | 0.9237 | 0.217 |
| 5 | 0.9099 | 0.9297 | 0.270 |
| 8 | 0.9099 | 0.9099 | 0.300 |

## Claim boundary

These values measure federated adaptation of one already selected checkpoint on the validation partition. They do not constitute federated training from scratch, a multi-estate privacy trial, or new locked-test evidence.
