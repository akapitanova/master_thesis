# SIMULATION OF HIGH-POWER LASER SPECTRA USING GENERATIVE MACHINE LEARNING

This project is a Master thesis by **Anna Kapitanova**, exploring the use of conditional diffusion-based generative models—specifically DDIM and EDM—for simulating the spectral output of the high-power L1-Allegra laser system. The goal was to create a digital twin capable of generating realistic spectral profiles based on experimental system parameters.

The project also includes a Bayesian optimization interface to support experimental guidance and analysis.

📄 The full thesis text is available as a PDF in the `text/` directory:  
**`text/kapitann_diploma_thesis.pdf`**

---

**Czech Technical University in Prague**  
**Faculty of Information Technology**  
© 2025 Bc. Anna Kapitánová. All rights reserved.

This thesis is school work as defined by the Copyright Act of the Czech Republic.  
It has been submitted at Czech Technical University in Prague, Faculty of Information Technology.  
The thesis is protected by the Copyright Act and its usage without the author’s permission is prohibited (with exceptions defined by the Act).

**Citation of this thesis**:  
Kapitánová Anna. *Simulation of high-power laser spectra using generative machine learning*.  
Master’s thesis. Czech Technical University in Prague, Faculty of Information Technology, 2025.

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone git@github.com:akapitanova/master_thesis.git
cd master_thesis
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 📁 Project Structure

#### 🔍 View Available Arguments

```bash
master\_thesis/
├── data/                  # Processed datasets (CSV, NPY)
├── preprocessing/         # Notebooks for cleaning, normalizing, and splitting data
├── src/                   # Source code for models, training, and evaluation
│   ├── models/            # Saved best models for EDM and DDIM
│   ├── predictions/       # Generated predictions of the best models
│   ├── *.py               # Core training and utility scripts
│   └── *.ipynb            # Evaluation and UI notebooks
```

## 🚀 Training

You can train both the EDM and DDIM diffusion models using command-line scripts.

### ✅ EDM Training Example

```bash
python src/train_edm.py \
  --run_name test \
  --epochs 30 \
  --batch_size 16 \
  --device cuda:0 \
  --dropout_rate 0.05 \
  --noise_steps 20 \
  --cfg_scale 3 \
  --cfg_scale_train 0.0 \
  --data_path data/train_data_stg7_norm.csv \
  --val_data_path data/val_data_stg7_clipped.csv \
  --sample_spectrum_path data/sample_spectrum_stgF.csv \
  --settings_dim 3 \
  --lr 1e-3
```
#### 🔍 View Available Arguments
```bash
python src/train_edm.py --help
```


### ✅ DDIM Training Example

```bash
python src/train_ddim.py \
  --run_name test \
  --epochs 30 \
  --batch_size 16 \
  --device cuda:0 \
  --dropout_rate 0.05 \
  --noise_steps 1000 \
  --beta_start 1e-4 \
  --beta_end 0.02 \
  --cfg_scale 3 \
  --cfg_scale_train 0.0 \
  --section_counts 40 \
  --data_path data/train_data_stg7_norm.csv \
  --val_data_path data/val_data_stg7_clipped.csv \
  --sample_spectrum_path data/sample_spectrum_stgF.csv \
  --settings_dim 3 \
  --lr 1e-3
```

#### 🔍 View Available Arguments
```bash
python src/train_ddim.py --help
```

## 📊 Results Visualizations
See the evaluated trained models using these notebooks in `src/`:

- `evaluate_edm.ipynb` — EDM validation results  
- `evaluate_ddim.ipynb` — DDIM validation results  
- `evaluate_edm_test_data.ipynb` — Best model evaluated on test set  

Predictions are stored in `src/predictions/<run_name>/`.

## 👤 UI Tools

Interactive Jupyter notebooks for experimentation:

- `Bayesian_optimization_ui.ipynb` — Tune model parameters using Bayesian optimization  
- `Spectrum_prediction.ipynb` — Generate new spectra using trained models

## ✨ Acknowledgements

Diffusion model techniques used in this project are inspired by the **DDIM** and **EDM** frameworks.

The implementation of the EDM model is based on the official NVIDIA code release:  
🔗 [https://github.com/NVlabs/edm](https://github.com/NVlabs/edm)
