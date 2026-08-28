# Exploratory Sparse Field-Note Modality Hierarchy

This robustness benchmark does not replace the standard modality ablation.
The internal test partition is not pristine because it was observed during
iterative protocol development.

- Selected text-token keep rate: `50%`
- Validation-selected reliability rule: `confidence < 0.6` with routes `[0, 0, 1, 0, 0]`
- Fixed-test hierarchy achieved: `True`

| System | Accuracy | Macro-F1 | Correct / 75 |
|---|---:|---:|---:|
| Sparse text only | 64.00% | 0.5798 | 48/75 |
| Enhanced image only | 72.00% | 0.6759 | 54/75 |
| Text + image fusion | 78.67% | 0.6472 | 59/75 |
| Proposed reliability-aware multimodal | 81.33% | 0.6679 | 61/75 |

Text sparsity and routing were selected on validation only. The final rule does not use fixed-test labels. Because this partition was observed during iterative development, it is an internal development test rather than pristine external evidence.
