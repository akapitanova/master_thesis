# SIMULATION OF HIGH-POWER LASER SPECTRA USING GENERATIVE MACHINE LEARNING


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

kapitann/
├── data/                  # Processed datasets (CSV, NPY)
├── preprocessing/         # Notebooks for cleaning, normalizing, and splitting data
├── src/                   # Source code for models, training, and evaluation
│   ├── models/            # Saved best models for EDM and DDIM
│   ├── predictions/       # Generated predictions of the best models
│   ├── *.py               # Core training and utility scripts
│   └── *.ipynb            # Evaluation and UI notebooks


## Training

You can train both the EDM and DDIM diffusion models using command-line scripts.

### EDM Training Example

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


### DDIM Training Example

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



