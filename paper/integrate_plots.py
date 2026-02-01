#!/usr/bin/env python3
"""
Script to integrate generated plots from FarmFederate training into the LaTeX paper.
Run this after training completes to copy plots and generate figure LaTeX code.
"""

import os
import shutil
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
PLOTS_SOURCE = PROJECT_ROOT / "output" / "plots"
FIGURES_DEST = SCRIPT_DIR / "figures"

# Expected plots from FarmFederate training
EXPECTED_PLOTS = {
    # Model comparison plots
    "model_comparison_f1.png": "F1 Score comparison across all model architectures",
    "model_comparison_accuracy.png": "Accuracy comparison across all model architectures",
    "model_type_comparison.png": "Performance comparison by model type (LLM, ViT, VLM)",

    # Training curves
    "training_curves_llm.png": "Training curves for LLM models",
    "training_curves_vit.png": "Training curves for ViT models",
    "training_curves_vlm.png": "Training curves for VLM fusion models",

    # Confusion matrices
    "confusion_matrix_bert.png": "Confusion matrix for BERT model",
    "confusion_matrix_vit_base.png": "Confusion matrix for ViT-Base model",
    "confusion_matrix_vlm_unified_io.png": "Confusion matrix for Unified-IO VLM",

    # Class distribution
    "class_distribution.png": "Class distribution across datasets",
    "prediction_distribution.png": "Prediction distribution analysis",

    # Federated learning
    "federated_convergence.png": "Federated learning convergence across rounds",
    "client_performance.png": "Per-client model performance",

    # SOTA comparison
    "sota_comparison_radar.png": "Radar chart comparing against SOTA methods",
    "sota_comparison_bar.png": "Bar chart comparing against SOTA methods",

    # Ablation
    "ablation_imbalance.png": "Ablation study on class imbalance techniques",
    "diversity_loss_effect.png": "Effect of diversity loss on class collapse",
}


def setup_figures_directory():
    """Create figures directory if it doesn't exist."""
    FIGURES_DEST.mkdir(parents=True, exist_ok=True)
    print(f"Figures directory: {FIGURES_DEST}")


def copy_plots():
    """Copy plots from training output to paper figures directory."""
    if not PLOTS_SOURCE.exists():
        print(f"Warning: Plots source directory not found: {PLOTS_SOURCE}")
        print("Run FarmFederate_Colab.py first to generate plots.")
        return []

    copied = []
    for plot_file in PLOTS_SOURCE.glob("*.png"):
        dest = FIGURES_DEST / plot_file.name
        shutil.copy2(plot_file, dest)
        copied.append(plot_file.name)
        print(f"Copied: {plot_file.name}")

    # Also copy PDFs and SVGs if available
    for ext in ["*.pdf", "*.svg"]:
        for plot_file in PLOTS_SOURCE.glob(ext):
            dest = FIGURES_DEST / plot_file.name
            shutil.copy2(plot_file, dest)
            copied.append(plot_file.name)
            print(f"Copied: {plot_file.name}")

    return copied


def generate_latex_figures(copied_plots):
    """Generate LaTeX figure code for copied plots."""
    latex_code = []

    latex_code.append("% Auto-generated figure includes for FarmFederate paper")
    latex_code.append("% Add this to your main.tex in the Results section\n")

    for plot_name, description in EXPECTED_PLOTS.items():
        if plot_name in copied_plots or plot_name.replace(".png", ".pdf") in copied_plots:
            # Use PDF if available, otherwise PNG
            if plot_name.replace(".png", ".pdf") in copied_plots:
                file_to_use = plot_name.replace(".png", ".pdf")
            else:
                file_to_use = plot_name

            label = plot_name.replace(".png", "").replace("_", "-")

            latex_code.append(f"""
\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=\\columnwidth]{{figures/{file_to_use}}}
\\caption{{{description}.}}
\\label{{fig:{label}}}
\\end{{figure}}
""")

    return "\n".join(latex_code)


def generate_results_appendix():
    """Generate a complete appendix with all plots."""
    appendix = []

    appendix.append("""
% ============================================================================
% APPENDIX: All Generated Plots
% ============================================================================
\\appendix
\\section{Complete Experimental Results}
\\label{sec:appendix}

This appendix contains all plots generated during the FarmFederate experiments.

\\subsection{Model Performance Comparison}
""")

    # Group plots by category
    categories = {
        "Model Comparison": ["model_comparison", "model_type"],
        "Training Curves": ["training_curves"],
        "Confusion Matrices": ["confusion_matrix"],
        "Class Distribution": ["class_distribution", "prediction_distribution"],
        "Federated Learning": ["federated", "client"],
        "SOTA Comparison": ["sota_comparison"],
        "Ablation Studies": ["ablation", "diversity_loss"],
    }

    for category, prefixes in categories.items():
        appendix.append(f"\n\\subsection{{{category}}}\n")

        for plot_name, description in EXPECTED_PLOTS.items():
            if any(plot_name.startswith(p) for p in prefixes):
                label = plot_name.replace(".png", "").replace("_", "-")
                appendix.append(f"""
\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=0.9\\columnwidth]{{figures/{plot_name}}}
\\caption{{{description}.}}
\\label{{fig:appendix-{label}}}
\\end{{figure}}
""")

    return "\n".join(appendix)


def main():
    print("=" * 60)
    print("FarmFederate Paper - Plot Integration Script")
    print("=" * 60)

    # Setup
    setup_figures_directory()

    # Copy plots
    print("\nCopying plots from training output...")
    copied = copy_plots()

    if not copied:
        print("\nNo plots found to copy.")
        print("Please run FarmFederate_Colab.py first to generate training plots.")
        return

    print(f"\nCopied {len(copied)} plot files.")

    # Generate LaTeX code
    print("\nGenerating LaTeX figure code...")
    latex_code = generate_latex_figures(copied)

    # Save to file
    latex_file = SCRIPT_DIR / "figures_include.tex"
    with open(latex_file, "w") as f:
        f.write(latex_code)
    print(f"Saved LaTeX figure code to: {latex_file}")

    # Generate appendix
    appendix_code = generate_results_appendix()
    appendix_file = SCRIPT_DIR / "appendix_plots.tex"
    with open(appendix_file, "w") as f:
        f.write(appendix_code)
    print(f"Saved appendix code to: {appendix_file}")

    print("\n" + "=" * 60)
    print("Integration complete!")
    print("=" * 60)
    print("\nTo include figures in main.tex, add:")
    print("  \\input{figures_include}")
    print("\nTo include appendix, add before \\end{document}:")
    print("  \\input{appendix_plots}")


if __name__ == "__main__":
    main()
