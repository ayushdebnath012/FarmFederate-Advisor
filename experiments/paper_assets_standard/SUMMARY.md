# FarmFederate ablation readout

## Data actually used

- paired samples: 1000 (train 700, val 150)
- real images before synthetic fill: 200
- class counts: [212, 199, 204, 182, 203]
- pairing mode: **label_matched_resample**
- image-text pairs are label-matched resamples, not genuine co-observations; fusion results must be read with that caveat

## E1 alpha sweep -- corrected vs legacy partitioner

- alpha=0.1 -- corrected: TV=0.571 F1=0.715 | legacy: TV=0.303 F1=0.909
- alpha=0.5 -- corrected: TV=0.392 F1=0.757 | legacy: TV=0.210 F1=0.879
- alpha=1 -- corrected: TV=0.339 F1=0.809 | legacy: TV=0.092 F1=0.886
- alpha=10 -- corrected: TV=0.108 F1=0.848 | legacy: TV=0.039 F1=0.843

- Corrected partitioner: F1 varies by 0.133 across the alpha range. This is the number that supports (or refuses to support) a robustness-to-non-IID claim.
- Legacy partitioner: F1 varies by 0.067. If this is near zero while TV is also near zero, the flatness is an artefact of the splitter, not evidence of robustness. Do not cite legacy-split numbers as non-IID results.

## E2 client-count scalability

- K=2: Macro F1 0.887
- K=3: Macro F1 0.814
- K=5: Macro F1 0.832
- K=10: Macro F1 0.604
- Read the drop from smallest to largest K as the cost of splitting fixed data across more sites, not as a property of the aggregation rule.

## E3 anti-collapse components

- alpha=0.1 Both (full system): F1=0.651, diversity=1.00
- alpha=0.1 -- diversity loss: F1=0.561, diversity=1.00
- alpha=0.1 -- balanced sampler: F1=0.771, diversity=1.00
- alpha=0.1 Neither: F1=0.736, diversity=1.00
- alpha=1 Both (full system): F1=0.906, diversity=1.00
- alpha=1 -- diversity loss: F1=0.919, diversity=1.00
- alpha=1 -- balanced sampler: F1=0.813, diversity=1.00
- alpha=1 Neither: F1=0.840, diversity=1.00

- IMPORTANT: the 'Neither' arm did NOT collapse (diversity stayed high without either component). The anti-collapse stack is then not doing the work the paper attributes to it on this data. Weaken the claim to match the measurement rather than re-running until it agrees.

## E4 warm start vs cold start

- warm 0.852 vs cold 0.809 (gap 0.043)

## E5 fusion variance

- gated: 0.890 +/- 0.010 over 2 seeds
- attention: 0.860 +/- 0.001 over 2 seeds
- concat: 0.809 +/- 0.005 over 2 seeds

## E6 communication cost

- concat: 59.5 MiB per client per round
- attention: 60.5 MiB per client per round
- gated: 60.3 MiB per client per round

## E8 matched-setting baselines

- alpha=0.1: local_only 0.279, fedavg 0.715, fedprox 0.706, scaffold 0.265, fedbn 0.572
- alpha=1: local_only 0.607, fedavg 0.809, fedprox 0.852, scaffold 0.255, fedbn 0.564

- This is the comparison reviewers asked for, and the only one in the paper run under matched conditions. Two readings matter: how far every federated arm sits above local-only (that is what federation buys), and whether the drift-correcting rules separate from FedAvg at low alpha.
- At alpha=0.1 nothing beat FedAvg by a clear margin. That is a legitimate, reportable result: state that the simplest rule sufficed at this scale.
