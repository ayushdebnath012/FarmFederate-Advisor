# Sparse Field-Note Modality Hierarchy

This robustness benchmark does not replace the standard modality ablation.

- Selected text-token keep rate: `55%`
- Validation-selected reliability rule: `confidence < 0.5` with routes `[1, 0, 0, 0, 0]`
- Fixed-test hierarchy achieved: `False`

| System | Accuracy | Macro-F1 | Correct / 75 |
|---|---:|---:|---:|
| Sparse text only | 72.00% | 0.6503 | 54/75 |
| Enhanced image only | 72.00% | 0.6759 | 54/75 |
| Text + image fusion | 78.67% | 0.7224 | 59/75 |
| Proposed reliability-aware multimodal | 80.00% | 0.7373 | 60/75 |

Text sparsity and routing were selected on validation only. The final rule does not use fixed-test labels. Because this partition was observed during iterative development, it is an internal development test rather than pristine external evidence.
