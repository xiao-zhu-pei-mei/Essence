# Parkinson's Disease (PD) Biomarker Discovery

An end-to-end pipeline for cerebrospinal-fluid (CSF) proteomics-based biomarker discovery.

---

## Project Workflow

```mermaid
graph LR
A[Raw CSF DIA/PRM Files] -->|process_data.py| B[Cleaned & Normalised Matrix]
B -->|AE_train.py / AEno_train.py / ML_train.py / DL_train.py| C[Model Training & Evaluation]
C --> D[Trained Model Artifacts]
D -->|Plot/*.py| E[Figures & Performance Charts]
```

**Key steps**
1. **Data preprocessing**: `process_data.py` harmonises two DIA batches (`2020-11-20_CSF_HBS_DIA` & `2020-11-20_CSF_LCC_DIA`) and the supplementary `mmc2.xlsx` metadata.
2. **Feature selection** (AE): backward incremental selection identifies the optimal 35-protein signature.
3. **Model training**
   - `ML_train.py` – traditional ML algorithms.
   - `DL_train.py` – deep-learning architecture in PyTorch.
   - `AE_train.py` – ablation experiments.
   - `AEno_train.py` – ablation experiments **without** core layers for control.
4. **Visualisation**: scripts inside `Plot/` generate publication-ready figures.

---

## Usage

### Detailed Commands
1. **Pre-process data**
   ```bash
   python process_data.py
   ```
2. **Train models** (examples)
   ```bash
   # Machine learning
   python ML_train.py

   # Deep learning
   python DL_train.py

   # Ablation with incremental feature selection
   python AE_train.py

   # Ablation without core layers
   python AEno_train.py
   ```
3. **Generate plots**
   ```bash
   python Plot/ML_IFS.py
   ```
---
