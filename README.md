# Brain Tumor Classification with Vision Transformers (ViT)

This repository contains the training notebook and related files for a Vision Transformer model used to classify brain MRI images into five categories:
- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor
- Unknown

The model is deployed on the MedScanAI platform where users can upload MRI scans and receive predictions.

Live Platform: [https://www.medscanai.net](https://www.medscanai.net/)  
Pretrained Model: [https://huggingface.co/itistamtran/vit_brain_tumor_multiclass_v2](https://huggingface.co/itistamtran/vit_brain_tumor_multiclass_v2)  

---

## Repository Contents

| File | Description |
|------|------------|
| `Brain_Tumor_Detected_ViT_22.ipynb` | Full training notebook including preprocessing, training, evaluation, and inference |
| `model_links.md` | External resource links such as model weights and documentation |
| `requirements.txt` | Python dependencies |

Large files such as trained model weights, datasets, and logs are hosted externally and not included in this repository.

---

## Dataset

The primary dataset used in this project is sourced from Kaggle:

**Brain Tumor MRI Dataset (Masoud Nickparvar)**  
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

An additional set of images was collected for the "Unknown" class to improve out-of-distribution robustness. The dataset is not included in this repository due to size and licensing restrictions.

---

## Model

The final fine-tuned model is publicly hosted on Hugging Face:

https://huggingface.co/itistamtran/vit_brain_tumor_multiclass_v2

Base model used for fine-tuning:  
`google/vit-base-patch16-224-in21k`

This model was trained using PyTorch and the Hugging Face Transformers library.

---

## Performance Summary

| Class | Precision | Recall | F1-score |
|-------|----------|--------|----------|
| Glioma | 0.97 | 0.98 | 0.98 |
| Meningioma | 0.97 | 0.96 | 0.97 |
| No Tumor | 0.99 | 1.00 | 0.99 |
| Pituitary | 0.99 | 0.99 | 0.99 |
| Unknown | 1.00 | 0.99 | 1.00 |

**Overall accuracy:** 98 percent

---

## Running the Notebook

### 1. Clone the repository

```bash
git clone https://github.com/itistamtran/brain-tumor-vit.git
cd brain-tumor-vit
```

### 2. Launch Jupyter Notebook
```bash
jupyter notebook
```
