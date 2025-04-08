# ROVER_MRI

This repository contains the implementation of **Rotating-view super-resolution (ROVER)-MRI reconstruction**, developed based on two ISMRM abstracts:

- *Rapid Whole Brain 180µm Mesoscale In-vivo T2w Imaging*, ISMRM 2025 (oral)
- *Rotating-view super-resolution (ROVER)-MRI reconstruction using tailored Implicit Neural Network*, ISMRM 2024 (oral power pitch)


## 📂 Directory Structure

```
ROVER_MRI/
├── configs/              # Experiment and training configurations
├── fda/                  # Domain adaptation and feature alignment modules
├── bumonkey_hash_v8_*.py # Main training & testing scripts
├── util_args_*.py        # Argument parsers for different experiments
├── utils.py              # Common utility functions
├── README.md             # Project documentation (this file)
```

## 🚀 Getting Started

### Prerequisites

- Python ≥ 3.8
- PyTorch ≥ 1.11 (tested with 1.13.1 + CUDA 11.7)
- NumPy, SciPy, nibabel, tqdm, etc.

Install dependencies (optional):

```bash
pip install -r requirements.txt
```

### Training

Modify the config file in `configs/` and run:

```bash
python bumonkey_hash_v8_noastype.py --config configs/your_config.yaml
```

### Evaluation

```bash
python bumonkey_hash_v8_test.py --config configs/your_test_config.yaml
```

## 📊 Results

We demonstrate high-quality reconstruction of whole-brain T2-weighted MRI using our enhanced ROVER-MRI framework.

Below is a comparison between the bicubic, LS-SRR, and our reconstructed high-resolution output (5× SR from 8 rotated views):

![Comparison](BUMonkey_Results/bumonkey.png)

## 📄 License

This project is released under the MIT License. See `LICENSE` for details.

---

