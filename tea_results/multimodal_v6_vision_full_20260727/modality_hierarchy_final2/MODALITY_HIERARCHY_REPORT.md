# Sparse Field-Note Modality Hierarchy

This robustness benchmark does not replace the standard modality ablation.

- Selected text-token keep rate: `45%`
- Validation-selected reliability rule: `confidence < 0.8` with routes `[1, 0, 1, 0, 0]`
- Fixed-test hierarchy achieved: `False`

| System | Accuracy | Macro-F1 | Correct / 75 |
|---|---:|---:|---:|
| Sparse text only | 65.33% | 0.5954 | 49/75 |
| Enhanced image only | 72.00% | 0.6759 | 54/75 |
| Text + image fusion | 77.33% | 0.6363 | 58/75 |
| Proposed reliability-aware multimodal | 74.67% | 0.6256 | 56/75 |

Text sparsity and routing were selected on validation only. The final rule does not use fixed-test labels. Because this partition was observed during iterative development, it is an internal development test rather than pristine external evidence.
