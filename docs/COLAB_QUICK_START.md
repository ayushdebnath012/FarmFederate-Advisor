# 🚀 Google Colab Quick Start Guide

**Run FarmFederate Training on Google Colab with Free GPU**

---

## 📋 Prerequisites

- Google account
- GitHub repository: `https://github.com/Solventerritory/FarmFederate-Advisor`
- Branch: `feature/multimodal-work`

---

## 🎯 Quick Start (2 Steps)

### **Step 1: Open in Colab**

Click this link to open the notebook directly in Colab:

👉 **[Open Comprehensive Training Notebook in Colab](https://colab.research.google.com/github/Solventerritory/FarmFederate-Advisor/blob/feature/multimodal-work/backend/Federated_LLM_ViT_VLM_Comprehensive_Training.ipynb)**

Or run the *one-click* launcher that prepares and validates the environment before running the full pipeline:

👉 **[One-Click FarmFederate Colab Launcher](https://colab.research.google.com/github/Solventerritory/FarmFederate-Advisor/blob/feature/multimodal-work/colab_run_farmfederate.ipynb)**

### **Step 2: Enable GPU & Run**

1. **Enable GPU:**
   - Click `Runtime` → `Change runtime type`
   - Select `T4 GPU` (or `V100` if available)
   - Click `Save`

2. **Run All Cells:**
   - Click `Runtime` → `Run all`
   - Wait 4-6 hours for complete training
   - Or run cells individually to test

---

## 📊 What Will Happen

### Training Pipeline
```
Install Dependencies (2-3 min)
    ↓
Load Datasets (1-2 min)
├─ AG News (text)
└─ PlantVillage (images)
    ↓
Train 17 Models (~4-6 hours)
├─ 9 LLM Models (text-based)
├─ 4 ViT Models (image-based)
└─ 4 VLM Models (multimodal)
    ↓
Generate 20+ Plots (~5 min)
    ↓
Create Report & Export Results
```

### Expected Runtime (with T4 GPU)
- **LLM Models:** ~2-3 hours (9 models)
- **ViT Models:** ~1-2 hours (4 models)
- **VLM Models:** ~1-2 hours (4 models)
- **Total:** ~4-6 hours

---

## 💡 Pro Tips

### 1. **Prevent Disconnection**

Colab disconnects after ~90 minutes of inactivity. To prevent this:

```javascript
// Run this in browser console (F12)
function KeepAlive() {
    console.log("Keeping session alive...");
    document.querySelector("colab-connect-button").click();
}
setInterval(KeepAlive, 60000); // Every 60 seconds
```

Or use this Python cell in the notebook:
```python
# Keep session alive
import IPython
display(IPython.display.Javascript('''
function KeepAlive() {
    console.log("Session alive");
}
setInterval(KeepAlive, 60000);
'''))
```

### 2. **Save Results to Google Drive**

Add this cell at the end to save results:
```python
from google.colab import drive
drive.mount('/content/drive')

# Copy results to Drive
!cp federated_training_results.json /content/drive/MyDrive/
!cp COMPREHENSIVE_REPORT.md /content/drive/MyDrive/
!cp -r plots /content/drive/MyDrive/farmfederate_plots
```

### 3. **Monitor GPU Usage**

Check GPU utilization:
```python
!nvidia-smi
```

### 4. **Reduce Training Time**

To train faster (for testing):
```python
# Modify FEDERATED_CONFIG in Section 2
FEDERATED_CONFIG = {
    'num_rounds': 3,      # Reduce from 10 to 3
    'local_epochs': 1,    # Reduce from 3 to 1
    # ... keep other settings
}
```

### 5. **Train Subset of Models**

To train only specific models:
```python
# In Section 7.1 (LLM training)
for model_name in LLM_MODELS[:2]:  # Only first 2 models
    # ... training code

# In Section 7.2 (ViT training)
for model_name in VIT_MODELS[:1]:  # Only first 1 model
    # ... training code

# In Section 7.3 (VLM training)
for model_name in VLM_MODELS[:1]:  # Only first 1 model
    # ... training code
```

---

## 🔧 Troubleshooting

### Issue 1: Out of Memory (OOM)

**Solution:** Reduce batch size
```python
FEDERATED_CONFIG['batch_size'] = 4  # Reduce from 8 to 4
```

### Issue 2: Session Disconnected

**Solution:** Restart from last checkpoint
- Training automatically saves checkpoints
- Simply restart and it will resume

### Issue 3: Dataset Download Fails

**Solution:** The notebook uses synthetic data as fallback
- AG News fails → Uses synthetic text
- PlantVillage fails → Uses synthetic images
- Training continues regardless

### Issue 4: Slow Training

**Solution 1:** Ensure GPU is enabled
```python
import torch
print(torch.cuda.is_available())  # Should print: True
```

**Solution 2:** Reduce model count (see "Train Subset of Models" above)

---

## 📥 Alternative: Manual Setup

If the direct link doesn't work, follow these steps:

### 1. Clone Repository
```python
!git clone -b feature/multimodal-work https://github.com/Solventerritory/FarmFederate-Advisor.git
%cd FarmFederate-Advisor/backend
```

### 2. Install Dependencies
```python
!pip install -q transformers>=4.40 datasets peft torch torchvision \
    pillow scikit-learn matplotlib seaborn numpy pandas \
    huggingface_hub accelerate sentencepiece protobuf timm einops scipy tqdm
```

### 3. Run Training
```python
# Open and run the notebook
%run Federated_LLM_ViT_VLM_Comprehensive_Training.ipynb
```

---

## 📊 Expected Outputs

### 1. JSON Results File
**File:** `federated_training_results.json`

**Content:**
```json
{
  "llm": {
    "model_name": {
      "history": {...},
      "final_f1": 0.XXX,
      "final_acc": 0.XXX
    }
  },
  "vit": {...},
  "vlm": {...}
}
```

### 2. Comprehensive Report
**File:** `COMPREHENSIVE_REPORT.md`

**Sections:**
- Executive Summary
- Model Performance (all 17 models)
- Baseline Comparisons (10 papers)
- Visualizations (20+ plots)
- Conclusions

### 3. Plots Directory
**Directory:** `plots/`

**Contains:**
- 20+ PNG files (300 DPI)
- Overall comparisons
- Convergence curves
- Per-class analysis
- Statistical tests

---

## 🎓 After Training

### 1. Generate Comparison Plots

After training completes, generate comparison analysis:

```python
# Run comparison framework
%cd /content/FarmFederate-Advisor/backend
!python run_comparison.py
```

**Output:** `plots/comparison/` with 8 additional plots

### 2. Download Results

```python
from google.colab import files

# Download JSON results
files.download('federated_training_results.json')

# Download report
files.download('COMPREHENSIVE_REPORT.md')

# Download all plots (zip first)
!zip -r plots.zip plots/
files.download('plots.zip')
```

### 3. View Plots in Colab

```python
from IPython.display import Image, display

# Display a plot
display(Image('plots/plot_01_overall_f1_comparison.png'))

# Display comparison plot
display(Image('plots/comparison/01_inter_category_comparison.png'))
```

---

## 📚 Additional Resources

### Documentation
- [COMPREHENSIVE_TRAINING_README.md](https://github.com/Solventerritory/FarmFederate-Advisor/blob/feature/multimodal-work/backend/COMPREHENSIVE_TRAINING_README.md) - Complete training guide
- [COMPARISON_FRAMEWORK_README.md](https://github.com/Solventerritory/FarmFederate-Advisor/blob/feature/multimodal-work/backend/COMPARISON_FRAMEWORK_README.md) - Comparison guide
- [DATASETS_USED.md](https://github.com/Solventerritory/FarmFederate-Advisor/blob/feature/multimodal-work/backend/DATASETS_USED.md) - Dataset documentation

### GitHub Repository
- **Main Repo:** https://github.com/Solventerritory/FarmFederate-Advisor
- **Branch:** `feature/multimodal-work`
- **Notebook:** `backend/Federated_LLM_ViT_VLM_Comprehensive_Training.ipynb`

---

## ✅ Checklist

Before starting:
- [ ] Google account logged in
- [ ] GPU enabled in Colab
- [ ] Notebook opened from GitHub link
- [ ] (Optional) Google Drive mounted for saving

During training:
- [ ] Monitor GPU usage with `!nvidia-smi`
- [ ] Keep session alive (use JavaScript snippet)
- [ ] Check progress in output cells

After training:
- [ ] Download results JSON
- [ ] Download plots
- [ ] Run comparison framework
- [ ] Review comprehensive report

---

## 🎉 That's It!

You're now ready to run comprehensive federated learning training on Google Colab with free GPU!

**Questions?** Check the documentation links above or GitHub issues.

---

**Direct Link to Notebook:**
👉 https://colab.research.google.com/github/Solventerritory/FarmFederate-Advisor/blob/feature/multimodal-work/backend/Federated_LLM_ViT_VLM_Comprehensive_Training.ipynb

**Happy Training! 🚀🌱**
